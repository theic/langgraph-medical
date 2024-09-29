import { AudioTranscriptLoader } from '@langchain/community/document_loaders/web/assemblyai';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

const systemMessage = {
  role: 'system',
  content:
    'You are a helpful assistant. You will receive the full transcription of an audio file as a message. This transcription is not a direct request, but rather the content of the audio. When you receive a transcription, first output the entire transcription, and then provide a summarized version of it. Analyze the language of the transcription and respond in the same language. If the language is unclear, default to English.',
};

const llm = new ChatOpenAI({
  model: 'gpt-4o-mini',
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

    console.debug('transcribeAudio Audio URL:', audioUrl);

    // Initialize the AudioTranscriptLoader with the extracted URL
    const audioTranscriptLoader = new AudioTranscriptLoader(
      { audio: audioUrl, language_detection: true, speech_model: 'best' },
      { apiKey: process.env.ASSEMBLYAI_API_KEY },
    );

    const docs = await audioTranscriptLoader.load();
    const transcription = docs[0].pageContent;
    console.debug('Transcription:', transcription);

    // Pass the transcription as a HumanMessage to the agent
    return { messages: [new HumanMessage(transcription)] };
  } catch (error) {
    console.error('Error transcribing audio:', error);
    return {
      messages: [
        new AIMessage("Sorry, I couldn't transcribe the audio file."),
      ],
    };
  }
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
