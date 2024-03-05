import bodyParser from "body-parser";
import express from "express";
import { BASE_ONION_ROUTER_PORT } from "../config";
import { REGISTRY_PORT } from "../config";
import axios from 'axios';

export async function simpleOnionRouter(nodeId: number) {
  const onionRouter = express();
  onionRouter.use(express.json());
  onionRouter.use(bodyParser.json());
  let lastReceivedEncryptedMessage: string | null = null;
  let lastReceivedDecryptedMessage: string | null = null;
  let lastMessageDestination: string | null = null;
  const publicKey: string | null = "test" + nodeId.toString();
  const privateKey: string | null = "azeaze" + nodeId.toString();


  axios.post("http://localhost:" + REGISTRY_PORT.toString() + "/registerNode",{
    nodeId : nodeId,
    pubKey : publicKey
  })



  onionRouter.get("/getLastReceivedEncryptedMessage", (req,res) => {
    res.status(200).json({ result: lastReceivedEncryptedMessage });
  })

  onionRouter.get("/getLastReceivedDecryptedMessage", (req,res) => {
    res.status(200).json({ result: lastReceivedDecryptedMessage });
  })

  onionRouter.get("/getLastMessageDestination", (req,res) => {
    res.status(200).json({ result: lastMessageDestination });
  })

  // TODO implement the status route
  // onionRouter.get("/status", (req, res) => {});
  onionRouter.get("/status", (req, res) => {
    res.send('live');
  });

  onionRouter.get("/getPrivateKey", (req,res) =>{
    res.status(200).json({ result: privateKey });
    //const payload = {
    //  result: privateKey
    //};
  
    //res.json(payload);
  

  })



  

  const server = onionRouter.listen(BASE_ONION_ROUTER_PORT + nodeId, () => {
    console.log(
      `Onion router ${nodeId} is listening on port ${
        BASE_ONION_ROUTER_PORT + nodeId
      }`
    );
  });

  return server;
}
