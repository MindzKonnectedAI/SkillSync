'use client'
import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import DialogS from './dialog';
import { Eye } from 'lucide-react';

const agents = [
  { name: 'Boolean agent' },
  { name: 'Github' },
  { name: 'Resume match' },
  { name: 'ATS' },
  { name: 'Reddit' },
];

const AgentTeam = () => {
  const [team, setTeam] = useState<{ name: string, orchestration?: string[] }[]>([]);
  const [availableAgents, setAvailableAgents] = useState(agents);
  const [open, setOpen] = useState({ open: false, type: "" })
  const [orchestration, setOrchestration] = useState<string[]>([])

  useEffect(() => {
    const storedSelection = JSON.parse(sessionStorage.getItem('selectedOptions') || '[]');
    const initialTeam = agents.filter(agent => storedSelection.includes(agent.name));
    setTeam(initialTeam);
    setAvailableAgents(agents.filter(agent => !storedSelection.includes(agent.name)));
  }, []);

  const addAgentToTeam = (agent: { name: string }) => {
    const newTeam = [...team, agent];
    setTeam(newTeam);
    setAvailableAgents(availableAgents.filter(a => a.name !== agent.name));
    sessionStorage.setItem('selectedOptions', JSON.stringify(newTeam.map(a => a.name)));
  };

  const removeAgentFromTeam = (agent: { name: string }) => {
    const newTeam = team.filter(a => a.name !== agent.name);
    setTeam(newTeam);
    setAvailableAgents([...availableAgents, agent]);
    sessionStorage.setItem('selectedOptions', JSON.stringify(newTeam.map(a => a.name)));
  };

  const viewOrchestrationHandle = (agent: string[]) => {
    setOpen({ open: true, type: "view-orchestration" });
    // setAvailableAgents([...availableAgents, agent]);
    // sessionStorage.setItem('selectedOptions', JSON.stringify(newTeam.map(a => a.name)));
    setOrchestration(agent)
  };

  console.log("team", team)

  return (
    <div className="p-8 bg-white min-h-screen text-black rounded-lg shadow-lg">
      <h1 className="text-4xl font-extrabold mb-8 text-center">Agent Team Manager</h1>
      <section className="mb-8">
        <div className='flex justify-between'>
          <h2 className="text-2xl font-bold mb-4">Current Agents in Team</h2>
          <div>
            <Button onClick={() => setOpen({ open: true, type: "orchestration" })}>Add agent orchestration</Button>
          </div>
        </div>
        {team.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {team.map((agent) => (
              // <Card 
              //   key={agent.name} 
              //   className="p-6 bg-gray-100 text-black rounded-2xl shadow-md hover:shadow-xl transition-shadow relative"
              // >
              //   <button 
              //     onClick={() => removeAgentFromTeam(agent)}
              //     className="absolute top-2 right-2 text-black text-lg font-bold bg-transparent px-2 py-1 rounded-full hover:bg-gray-200"
              //   >
              //     ✖
              //   </button>
              //   <CardContent className="flex flex-col justify-center items-center space-y-4">
              //     <p className="text-xl font-semibold capitalize">{agent.name}</p>
              //   </CardContent>
              // </Card>
              <Card
                key={agent.name}
                className="p-4 bg-gray-100 text-black rounded-lg shadow-md hover:bg-gray-200 cursor-pointer flex justify-between items-center"
              >
                <p className="text-lg font-semibold capitalize">{agent.name}</p>
                <div className='flex gap-2 items-center'>
                  {agent.orchestration && <Eye onClick={() => viewOrchestrationHandle(agent?.orchestration ?? [])} />}
                  <Button size="sm"
                    onClick={() => removeAgentFromTeam(agent)}
                    className="bg-black text-white px-4 py-2 rounded-lg"
                  >
                    Remove
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <p className="text-lg italic text-gray-500">No agents in the team yet.</p>
        )}
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Upgrade Add Agents to Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {availableAgents.map((agent) => (
            <Card
              key={agent.name}
              className="p-4 bg-gray-100 text-black rounded-lg shadow-md hover:bg-gray-200 cursor-pointer flex justify-between items-center"
            >
              <p className="text-lg font-semibold capitalize">{agent.name}</p>
              <button
                onClick={() => addAgentToTeam(agent)}
                className="bg-black text-white px-4 py-2 rounded-lg"
              >
                Add
              </button>
            </Card>
          ))}
        </div>
        <p className="mt-6 text-lg">Number of agents in team: <span className="font-bold">{team.length}</span></p>
      </section>
      {open.open && <DialogS open={open} setOpen={setOpen} team={team} setTeam={setTeam} orchestration={orchestration} />}
    </div>
  );
};

export default AgentTeam;
