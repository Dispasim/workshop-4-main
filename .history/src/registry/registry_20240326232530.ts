import bodyParser from "body-parser";
import express, { Request, Response } from "express";
import { REGISTRY_PORT } from "../config";

export type Node = { nodeId: number; pubKey: string };

export type RegisterNodeBody = {
  nodeId: number;
  pubKey: string;
};

export type GetNodeRegistryBody = {
  nodes: Node[];
};



export async function launchRegistry() {
  let registeredNodes: Node[] = [];
  const _registry = express();
  _registry.use(express.json());
  _registry.use(bodyParser.json());

 

  // TODO implement the status route
  // _registry.get("/status", (req, res) => {});*
  _registry.get("/status", (req, res) => {
    //res.status(200).json({ status: 'live' });
    res.send('live');
  });

  _registry.post("/registerNode", (req,res) => {
    const {nodeId, pubKey} = req.body

    const newNode: RegisterNodeBody = {
      nodeId: nodeId,
      pubKey: pubKey
    };

    registeredNodes.push(newNode);
   


  });

  _registry.get("/getNodeRegistry", (req,res) => {

  const payload = {
    nodes: registeredNodes
  };

  res.json(payload);

  });

  const server = _registry.listen(REGISTRY_PORT, () => {
    console.log(`registry is listening on port ${REGISTRY_PORT}`);
  });

  return server;
}
