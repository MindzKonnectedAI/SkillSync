import * as React from "react"

import { Button } from "@/components/ui/button"
import {
    Sheet,
    SheetClose,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
    SheetTrigger,
} from "@/components/ui/sheet"
import Image from "next/image"
import agent from "@/image/ai-agent.gif"
import github from "@/image/Github.gif"
import { Card } from '@/components/ui/card';
import { useState } from "react"
import aiagent from "@/image/aiagent.png"

interface DialogSProps {
    sheetOpen: { open: boolean; type: string };
    agents: { name: string, add:boolean, close?: boolean }[];
    setSheetOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setAgents: React.Dispatch<React.SetStateAction<{ name: string, add:boolean, close?: boolean }[]>>;
    team?: { name: string }[];
}


export default function AgentSheet({ sheetOpen, setSheetOpen, agents, setAgents, team }: DialogSProps) {

    // const agents = [
    //     { name: 'Boolean agent' },
    //     { name: 'Github' },
    //     { name: 'Profile checker' },
    //     { name: 'ATS' },
    //     { name: 'Reddit' },
    // ];

    const [availableAgents, setAvailableAgents] = useState(team ?? []);

    const agentSelect = (name: string) => {
        setAvailableAgents((prev) => prev.filter(agent => agent.name !== name)); // Remove the selected agent
        setAgents((prev) => [...prev, { "name": name, add: true }]); // Add the selected agent to agents
        setSheetOpen({ open: false, type: "" });
    }

    const handleonOpenChange = (open: boolean) => {
        setSheetOpen(prev => ({...prev, open }));
        // if (agents && agents.length > 0) {
        //     setAgents((prev) => [...prev, { "name": team[team.length - 1].name, add: true }]); // Add the selected agent to agents
        // }
    }
    return (
        <Sheet open={sheetOpen.open} onOpenChange={(open: boolean) => handleonOpenChange(open)} >
            <SheetTrigger asChild>
                <Button type="submit" size="sm" variant="outline" className="relative px-6 py-3 text-black bg-white shadow-sm rounded-full border-2 border-transparent overflow-hidden group">
                    {/* <span className="absolute inset-0 rounded-full border-2 border-zinc-500 animate-pulse"></span>
                    <span className="relative z-10">Agents</span> */}
                </Button>
            </SheetTrigger>
            <SheetClose asChild>
                <Button type="submit" size="sm" variant="outline" className="relative px-6 py-3 text-black bg-white shadow-sm rounded-full border-2 border-transparent overflow-hidden group">
                    {/* <span className="absolute inset-0 rounded-full border-2 border-zinc-500 animate-pulse"></span> */}
                    <span className="relative z-10">close</span>
                </Button>
            </SheetClose>
            <SheetContent>
                <SheetHeader>
                    <SheetTitle>Agents</SheetTitle>
                    <SheetDescription>
                        <div className="grid gap-4">
                            {availableAgents.map((agent) => (
                                <Card onClick={() => agentSelect(agent.name)}
                                    key={agent.name}
                                    className="p-4 bg-gray-100 text-black rounded-sm hover:bg-gray-200 cursor-pointer flex items-center gap-3"
                                >
                                    <Image src={aiagent} alt="" width={30} height={30} />
                                    <p className="text-lg font-semibold capitalize">{agent.name}</p>
                                    {/* <button
                                        onClick={() => addAgentToTeam(agent)}
                                        className="bg-black text-white px-4 py-2 rounded-lg"
                                    >
                                        Add
                                    </button> */}
                                </Card>
                            ))}
                        </div>
                    </SheetDescription>
                </SheetHeader>
                {/* <SheetFooter>
          <SheetClose asChild className="mt-5">
            <Button type="submit">Close</Button>
          </SheetClose>
        </SheetFooter> */}
            </SheetContent>
        </Sheet>
    )
}
