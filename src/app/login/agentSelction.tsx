'use client'
import { cn } from "@/lib/utils"
import React, { useState } from 'react';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Lock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from "next/link"
import { useRouter } from "next/navigation"
import { BadgeInfo } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"


export default function AgentSelction() {
  const router = useRouter();
  const options: { label: string; value: string; defaultChecked: boolean; locked?: boolean, discription: string }[] = [
    // { label: 'Boolean (by default)', value: 'Boolean agent', defaultChecked: true, locked: true },
    { label: 'Boolean', value: 'Boolean agent', defaultChecked: true, discription: "An AI agent that dynamically generates precise Boolean queries based on user input and contextual data." },

    { label: 'GitHub', value: 'Github', defaultChecked: true, discription: "A GitHub agent that fetches and processes user data dynamically." },
    { label: 'Resume match', value: 'Resume match', defaultChecked: true, discription: "An AI agent that matches resumes with job descriptions efficiently." },
    { label: 'ATS', value: 'ATS', defaultChecked: true, discription: "An ATS agent that finds candidates from data sources based on job descriptions." },
    { label: 'Reddit', value: 'Reddit', defaultChecked: true, discription: "A Reddit agent that finds users based on specified criteria." },
  ];
  const [selectedOptions, setSelectedOptions] = useState(
    options.filter(option => option.defaultChecked).map(option => option.value)
  );

  const handleChange = (value: string, locked: boolean | undefined) => {
    if (locked) return; // Prevent unchecking locked options
    setSelectedOptions((prev) =>
      prev.includes(value)
        ? prev.filter((option) => option !== value)
        : [...prev, value]
    );
  };

  const handleNext = () => {
    console.log('Selected Options:', selectedOptions);
    sessionStorage.setItem('selectedOptions', JSON.stringify(selectedOptions));
    router.push('/talent-finder/');
  };
  return (
    <div className={cn("flex flex-col gap-6")} >
      <Card className="p-8">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Welcome</CardTitle>
          <CardDescription>
            Select your agent team
          </CardDescription>
        </CardHeader>
        <h2 className="text-xl font-bold mb-4">Select Options</h2>
        <CardContent>
          {options.map((option) => (
            <label key={option.value} className="flex items-center space-x-3">
              <div className="flex flex-col w-full">
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={selectedOptions.includes(option.value)}
                    onCheckedChange={() => handleChange(option.value, option.locked)}
                  />
                  <span className="text-base">{option.label}</span>
                </div>
                <div className="flex  justify-between">
                  <div className="text-base pl-6 text-[0.9rem] text-zinc-400 flex justify-between">
                    {option.discription.slice(0, 47)}{option.discription.length > 50 && "....."}
                  </div>
                  <div>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className='cursor-pointer'>
                          <BadgeInfo className="cursor-pointer ml-auto" size={20} />
                        </TooltipTrigger>
                        <TooltipContent className='bg-white'>
                          {option.discription}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                  </div>
                </div>
              </div>
              {option.locked && <Lock className="w-4 h-4 text-gray-400" />}
            </label>
          ))}
        </CardContent>
        <Button onClick={handleNext} className="mt-4 w-full">Next</Button>
      </Card>
    </div >
  )
}


// import React, { useState } from 'react';
// import { Checkbox } from '@/components/ui/checkbox';
// import { Card, CardContent } from '@/components/ui/card';
// import { Lock } from 'lucide-react';
// import { Button } from '@/components/ui/button';

// const options = [
//   { label: 'Boolean (by default)', value: 'boolean', defaultChecked: true, locked: true },
//   { label: 'GitHub', value: 'github' },
//   { label: 'ATS', value: 'ats' },
//   { label: 'Reddit', value: 'reddit' },
// ];

// export default function MultiCheckbox() {
//   const [selectedOptions, setSelectedOptions] = useState(
//     options.filter(option => option.defaultChecked).map(option => option.value)
//   );

//   const handleChange = (value, locked) => {
//     if (locked) return; // Prevent unchecking locked options
//     setSelectedOptions((prev) =>
//       prev.includes(value)
//         ? prev.filter((option) => option !== value)
//         : [...prev, value]
//     );
//   };

//   const handleNext = () => {
//     console.log('Selected Options:', selectedOptions);
//   };

//   return (
//     <div className="max-w-md mx-auto p-4">
//       <Card className="rounded-2xl shadow-lg p-4">
//         <h2 className="text-xl font-bold mb-4">Select Options</h2>
//         <CardContent className="space-y-3">
//           {options.map((option) => (
//             <label key={option.value} className="flex items-center space-x-3">
//               <Checkbox
//                 checked={selectedOptions.includes(option.value)}
//                 onCheckedChange={() => handleChange(option.value, option.locked)}
//               />
//               <span className="text-base">{option.label}</span>
//               {option.locked && <Lock className="w-4 h-4 text-gray-400" />}
//             </label>
//           ))}
//         </CardContent>
//         <Button onClick={handleNext} className="mt-4 w-full">Next</Button>
//       </Card>
//     </div>
//   );
// }
