import { AudioTranscriptLoader } from '@langchain/community/document_loaders/web/assemblyai';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { doctorPatientDialog, patientCardTemplate } from 'constant.js';

const systemMessage = {
  role: 'system',
  content:
    `You are a medical assistant tasked with analyzing doctor-patient dialogues and generating structured patient cards. When you receive a transcription of a doctor-patient conversation, your job is to create a detailed patient card based on the information provided. Use the following example as a guide for structuring the patient card:

    Example of a doctor-patient dialogue:
    ${doctorPatientDialog}

    Patient Card Template:
    ${patientCardTemplate}

    After generating the patient card, be prepared to answer follow-up questions about the context of the dialogue. Analyze the language of the transcription and respond in the same language. If the language is unclear, default to English.
    Keep markdown syntax in the patient card.

    If the message request is not clear or no transcription is provided, ask for additional information or request an audio file for transcription. For example:
    "I'm sorry, but I need more information to assist you properly. Could you please provide more details about your request or share an audio file of the doctor-patient conversation for transcription?"
    `,
};

const llm = new ChatOpenAI({
  model: 'gpt-4o',
  temperature: 0,
});

// Modify the transcribeAudio function to extract the audio URL directly
const transcribeAudio = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  const content = lastMessage.content as string;

  console.debug('lastMessage', lastMessage);

  try {
    // Extract the audio URL from the message content
    const audioUrl = extractAudioUrl(content);

    // Initialize the AudioTranscriptLoader with the extracted URL
    const audioTranscriptLoader = new AudioTranscriptLoader(
      { audio: audioUrl, language_detection: true, speech_model: 'best', speaker_labels: true, speakers_expected: 2 },
      { apiKey: process.env.ASSEMBLYAI_API_KEY },
    );

    const docs = await audioTranscriptLoader.load();
    const transcriptionJson = docs[0].metadata.utterances?.map(utterance => ({
      speaker: utterance.speaker,
      text: utterance.text
    }));

    // Convert JSON to formatted text
    const formattedTranscription = formatTranscription(transcriptionJson);

    // Pass the formatted transcription as a HumanMessage to the agent
    return { messages: [new HumanMessage(formattedTranscription)] };
  } catch (error) {
    console.error('Error transcribing audio:', error);
    return {
      messages: [
        new AIMessage("Sorry, I couldn't transcribe the audio file."),
      ],
    };
  }
};

// Helper function to format the transcription
const formatTranscription = (transcription: Array<{ speaker: string; text: string }> | undefined): string => {
  if (!transcription) return "No transcription available.";

  return transcription.map(({ speaker, text }) => {
    return `${speaker}:\n  ${text.split('\n').join('\n  ')}`;
  }).join('\n\n');
};

const extractAudioUrl = (content: string): string => {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const urls = content.match(urlRegex);
  if (urls && urls.length > 0) {
    return urls[0];
  }
  throw new Error('No valid audio URL found in the message.');
};

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  const result = await llm.invoke([systemMessage, ...messages.slice(1)]);
  return { messages: [result] };
};

const shouldTranscribe = (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  const content = lastMessage.content as string;

  if (lastMessage._getType() === 'human' && content.includes('http')) {
    return 'transcribe';
  }
  return 'agent';
};

const workflow = new StateGraph(MessagesAnnotation)
  .addNode('transcribe', transcribeAudio)
  .addNode('agent', callModel)
  .addConditionalEdges(START, shouldTranscribe, ['transcribe', 'agent'])
  .addEdge('transcribe', 'agent')
  .addEdge('agent', END);

export const graph = workflow.compile();
