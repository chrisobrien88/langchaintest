// 1. Import necessary modules and libraries
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as dotenv from "dotenv";

// 2. Load environment variables
dotenv.config();

const checkAPIresponse = async () => {
  // checking api key has valid requests
  const apiKey = process.env.OPENAI_API_KEY;
  console.log("api key", apiKey);
  const apiEndpoint = "https://api.openai.com/v1/engines/davinci/completions";
  const payload = {
    prompt: "This is a test",
    max_tokens: 5,
    temperature: 0.9,
    top_p: 1,
    n: 1,
    stream: false,
    logprobs: null,
    stop: "\n",
  };

  const response = await fetch(apiEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(payload),
  });

  // Get the remaining rate limit and reset time from the response headers
  const remainingLimit = response.headers.get("x-ratelimit-remaining");
  const resetTime = response.headers.get("x-ratelimit-reset");

  console.log(response);
};

// checkAPIresponse();

// 3. Set up input data and paths
const txtFilename = "the_anatomy_of_drunkenness";
const question = "What are the causes of drunkenness?";
const txtPath = `./${txtFilename}.txt`;
const VECTOR_STORE_PATH = `${txtFilename}.index`;

fs.access(txtPath, fs.constants.R_OK | fs.constants.W_OK, (err) => {
  if (err) {
    console.error("File access error:", err);
    return;
  }

  console.log("File is readable and writable.");
});

// 4. Define the main function runWithEmbeddings
export const runWithEmbeddings = async () => {
  // 5. Initialize the OpenAI model with an empty configuration object
  const model = new OpenAI({});

  // 6. Check if the vector store file exists
  let vectorStore;
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    // 6.1 If it exists, load it into memory
    console.log("Vector exists");
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    // 6.2. If the vector store file doesn't exist, create it
    // 6.2.1. Read the input text file
    console.log("Reading txtPath file");
    const text = fs.readFileSync(txtPath, "utf-8");
    console.log("TxtPath file read");

    // 6.2.2. Create a RecursiveCharacterTextSplitter with a specified chunk size (embeddings endpoint in OpenAI has a limit of 2048 characters with each request)
    console.log("Creating text splitter");
    const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1000});
    console.log("Text splitter created");

    // 6.2.3. Split the input text into documents
    console.log("Creating documents");
    const docs = await textSplitter.createDocuments([text]);
    console.log("Documents:", docs[0]);
    console.log("Documents created");

    // 6.2.4. Create a new vector store from the documents using OpenAIEmbeddings
    try {
      console.log("Creating vector store");
      vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
      console.log("Vector store created");
    } catch (error) {
      console.log("Error creating vector store:", error);
      return;
    }

    // 6.2.5. Save the vector store to a file
    console.log("Saving vector store");
    await vectorStore.save(VECTOR_STORE_PATH);
    console.log("Vector store saved");
  }

  // 7. Create a RetrievalQAChain by passing the initialized OpenAI model and the vector store retriever
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  // 8. Define a function to handle rate limiting errors and retry the request
  const res = await chain.call({
    query: question,
  })

  // 9. Execute the retrieval question-answering process
  
    console.log("Answer:", ({ res }));
    
  
};

// 10. Run the main function
runWithEmbeddings();
