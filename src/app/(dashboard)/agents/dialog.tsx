'use client'
import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { useState } from 'react'
import { Textarea } from "@/components/ui/textarea"
import axios from 'axios'
import { SquarePlus, SquareX, CircleX, CirclePlus } from 'lucide-react';
import AgentSheet from "./AgentSheet"
import Image from "next/image"
import { Card } from "@/components/ui/card"
import aiagent from "@/image/aiagent.png"

interface ContentProps {
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setJDContent?: React.Dispatch<React.SetStateAction<string>>;
    findMatch?: () => void;
    orchestration?: string[];
    team?: { name: string, orchestration?: string[] }[];
    setTeam: React.Dispatch<React.SetStateAction<{ name: string, orchestration?: string[] }[]>>;
}
interface sheetStateProps {
    open: boolean,
    type: string
}
interface agentStateProps {
    name: string,
    add: boolean,
    close?: boolean
}

const ATS: React.FC<ContentProps> = ({ team, setTeam, setOpen }: ContentProps) => {
    const [sheetOpen, setSheetOpen] = useState<sheetStateProps>({ open: false, type: "" })
    const [agents, setAgents] = useState<agentStateProps[]>([])

    const addAgent = ({ name, add }: agentStateProps) => {
        // setConnect(value)
        setAgents(prev => {
            const existingAgent = prev.find(agent => agent.name === name);

            if (!existingAgent) {
                return [...prev, { name, add }];
            } else {
                return prev.map(agent =>
                    agent.name === name ? { ...agent, add } : agent
                );
            }
        });
        setSheetOpen({ open: true, type: "agent" })
        // setOpen(prev => ({ ...prev, open: false }))
    }

    const removeAgent = ({ name }: agentStateProps) => {
        setAgents(prev => {
            // Filter out the removed agent
            const updatedAgents = prev.filter(agent => agent.name !== name);

            // If there's at least one agent left, set `add: true` for the last one
            if (updatedAgents.length > 0) {
                updatedAgents[updatedAgents.length - 1] = {
                    ...updatedAgents[updatedAgents.length - 1],
                    add: true
                };
            }

            return updatedAgents;
        });

    }

    const handleCreate = () => {
        if (agents.length > 0) {
            const findFirstAgent = agents[0];

            setTeam(prev => {
                return prev.map((teamMember: { name: string; orchestration?: string[] }) =>
                    teamMember.name === findFirstAgent.name
                        ? { ...teamMember, orchestration: agents.map(agent => agent.name) }
                        : teamMember
                );

            });
        }

        setOpen({ open: false, type: "" });
    };

    return (
        <>{sheetOpen.open && <AgentSheet sheetOpen={sheetOpen} setSheetOpen={setSheetOpen} setAgents={setAgents} team={team} agents={agents} />}
            <div className={`flex items-center ${agents.length === 0 && "justify-center"} space-x-2 py-5`}>
                {agents.length === 0 && <SquarePlus className="cursor-pointer" size={40} onClick={() => setSheetOpen({ open: true, type: "agent" })} />}
                {agents.length > 0 &&
                    <div className="flex">
                        {agents.map((agent, index) => (
                            <div className="flex justify-center items-center relative" key={index}>
                                <Card
                                    
                                    className="bg-gray-100 text-black rounded-md hover:bg-gray-200 cursor-pointer flex items-center gap-2 p-2"
                                >
                                    <Image src={aiagent} alt="" width={30} height={30} />
                                    <p className="text-[12px] capitalize">{agent.name}</p>
                                    {agent.add && <CircleX className="cursor-pointer absolute right-[41px] top-[-8px]" size={20} onClick={() => removeAgent({ "name": agent.name, add: false })} />}
                                </Card>
                                <div className="flex justify-center items-center">
                                    <div className="w-[30px] h-[1px] bg-zinc-400"></div>
                                    {agent.add && <CirclePlus className="cursor-pointer" size={20} onClick={() => addAgent({ "name": agent.name, add: false })} />}
                                </div>
                            </div>
                        ))}
                    </div>
                }
            </div>
            <DialogFooter className="sm:justify-start">
                <Button type="submit" size="sm" className="px-3" onClick={handleCreate}>
                    Create
                </Button>
            </DialogFooter>
        </>
    )
}

const Orchestration: React.FC<ContentProps> = ({ orchestration }: ContentProps) => {

    return (
        <>
           <div className={`flex items-center space-x-2 py-5`}>
                {orchestration && orchestration.length > 0 &&
                    <div className="flex">
                        {orchestration.map((agent, index) => (
                            <div className="flex justify-center items-center relative" key={index}>
                                <Card
                                  
                                    className="bg-gray-100 text-black rounded-md hover:bg-gray-200 cursor-pointer flex items-center gap-2 p-2"
                                >
                                    <Image src={aiagent} alt="" width={30} height={30} />
                                    <p className="text-[12px] capitalize">{agent}</p>
                                    {/* {agent.add && <CircleX className="cursor-pointer absolute right-[41px] top-[-8px]" size={20} onClick={() => removeAgent({ "name": agent.name, add: false })} />} */}
                                </Card>
                                <div className="flex justify-center items-center">
                                    {orchestration[orchestration.length -1] !== agent && <div className="w-[30px] h-[1px] bg-zinc-400"></div>}
                                    {/* {agent.add && <CirclePlus className="cursor-pointer" size={20} onClick={() => addAgent({ "name": agent.name, add: false })} />} */}
                                </div>
                            </div>
                        ))}
                    </div>
                }
            </div>
        </>
    )
}

interface DialogSProps {
    open: { open: boolean; type: string };
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    team?: { name: string, orchestration?: string[] }[];
    setTeam: React.Dispatch<React.SetStateAction<{ name: string, orchestration?: string[] }[]>>;
    orchestration: string[];
}
// interface DialogProps {
//     open: { open: boolean, type: string };
//     setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
//     team: { name: string, orchestration?: string[] }[];
//     orchestration: string[];
//   }
export default function dialog({ open, setOpen, team, setTeam, orchestration }: DialogSProps) {

    return (
        <Dialog open={open.open} onOpenChange={(isOpen: boolean) => setOpen(prev => ({ ...prev, open: isOpen }))}>
            <DialogContent className="max-w-fit min-w-[500px]">
                <DialogHeader>
                    <DialogTitle>
                        {open.type === "orchestration" && "Create agent orchestration"}
                        {open.type === "view-orchestration" && "Orchestration"}
                    </DialogTitle>
                </DialogHeader>
                {open.type === "orchestration" && <ATS setOpen={setOpen} team={team} setTeam={setTeam} />}
                {open.type === "view-orchestration" && <Orchestration setOpen={setOpen} team={team} setTeam={setTeam} orchestration={orchestration} />}
            </DialogContent>
        </Dialog>
    )
}
