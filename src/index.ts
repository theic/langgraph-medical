import { AudioTranscriptLoader } from '@langchain/community/document_loaders/web/assemblyai';
import { AIMessage } from '@langchain/core/messages';
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
    'You are an audio transcription tool. Your task is to accurately transcribe\n' +
    'audio content provided to you. You should maintain the original meaning\n' +
    'and context of the audio, capturing spoken words, relevant sound effects,\n' +
    'and speaker changes where applicable. Ensure the transcription is clear,\n' +
    'well-formatted, and easy to read.',
};

const llm = new ChatOpenAI({
  model: 'gpt-4o',
  temperature: 0,
});

const transcribeAudio = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  console.debug('lastMessage', lastMessage);

  const content = lastMessage.content as string;
  
  if (lastMessage._getType() === 'human' && content.includes('http')) {
    try {
      // Extract the audio URL from the message content
      const audioUrl = extractAudioUrl(content);

      // Initialize the AudioTranscriptLoader with the extracted URL
      const audioTranscriptLoader = new AudioTranscriptLoader(
        {
          audio: audioUrl,
        },
        {
          apiKey: process.env.ASSEMBLYAI_API_KEY,
        }
      );

      const docs = await audioTranscriptLoader.load();
      const transcription = docs[0].pageContent;
      console.debug('Transcription:', transcription);
      return { messages: [new AIMessage(`Transcription: ${transcription}`)] };
    } catch (error) {
      console.error('Error transcribing audio:', error);
      return { messages: [new AIMessage('Sorry, I couldn\'t transcribe the audio file.')] };
    }
  }
  
  return { messages: [] };
};

// Add a utility function to extract the audio URL from the message
const extractAudioUrl = (content: string): string => {
  // Implement a method to extract a valid audio URL from the content
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const urls = content.match(urlRegex);
  if (urls && urls.length > 0) {
    return urls[0];
  }
  throw new Error('No valid audio URL found in the message.');
};

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const result = await llm.invoke([systemMessage, ...messages]);
  return { messages: [result] };
};

const shouldTranscribe = (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  console.debug('lastMessage', lastMessage);

  const content = lastMessage.content as string;

  if (lastMessage._getType() === 'human' && content.includes('http')) {
    return 'transcribe';
  }
  return 'agent';
};

const workflow = new StateGraph(MessagesAnnotation)
  .addNode('transcribe', transcribeAudio)
  .addNode('agent', callModel)
  .addEdge(START, 'transcribe')
  .addEdge(START, 'agent')
  .addConditionalEdges(
    START,
    shouldTranscribe
  )
  .addEdge('transcribe', 'agent')
  .addEdge('agent', END);

export const graph = workflow.compile();
