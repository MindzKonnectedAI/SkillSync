import * as React from "react"

import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import Image from "next/image"
import agent from "@/image/ai-agent.gif"
// import github from "@/image/Github.gif"

export default function AgentActivitySheet({ data }: { data: any | null }) {

  return (
    <Sheet >
      <SheetTrigger asChild>
        <Button type="submit" size="sm" variant="outline" className="relative px-6 py-3 text-black bg-white shadow-sm rounded-full border-2 border-transparent overflow-hidden group">
          <span className="absolute inset-0 rounded-full border-2 border-zinc-500 animate-pulse"></span>
          <span className="relative z-10">Check activity</span>
        </Button>
      </SheetTrigger>
      <SheetContent style={{ maxWidth: '500px' }}>
        <SheetHeader>
          <SheetTitle>Agent activity</SheetTitle>
          <SheetDescription className="overflow-y-scroll">
            <div className={`flex flex-col gap-1 p-2 rounded-t-5  h-[90dvh] `}>
              {data.map((res: any, index: any) => (
                <div className="flex gap-2 bg-muted p-2" key={index} >
                  <div className="flex justify-center items-center rounded-full w-[40px] h-[40px]">
                    <Image src={agent} alt="" className="rounded-full min-w-[40px] min-h-[40px]" />
                  </div>
                  <div className="">
                    <p className="text-gray-500 text-md">{res && Object.keys(res)}</p>
                    <div className="text-gray-500 text-sm break-words" style={{width: "350px"}}>{res && JSON.stringify(res)}</div>
                  </div>
                  {/* {msg.agent === "github-agent" && <div className="flex items-center gap-2">
                    <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                      <Image src={github} alt="" className="rounded-full" />
                    </div>
                    <div className="">
                      <p className="text-gray-500 text-md">Github</p>
                      <p className="text-gray-500 text-sm">{msg.message}</p>
                    </div>
                  </div>} */}
                </div>))}
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
// import * as React from "react"

// import { Button } from "@/components/ui/button"
// import {
//   Sheet,
//   SheetContent,
//   SheetDescription,
//   SheetHeader,
//   SheetTitle,
//   SheetTrigger,
// } from "@/components/ui/sheet"
// import Image from "next/image"
// import agent from "@/image/ai-agent.gif"
// import github from "@/image/Github.gif"

// export default function AgentActivitySheet({ displayedMessages }: { displayedMessages: { agent: string, message: string, data:any }[] }) {
//   console.log("displayedMessages", displayedMessages)
//   return (
//     <Sheet>
//       <SheetTrigger asChild>
//         <Button type="submit" size="sm" variant="outline" className="relative px-6 py-3 text-black bg-white shadow-sm rounded-full border-2 border-transparent overflow-hidden group">
//           <span className="absolute inset-0 rounded-full border-2 border-zinc-500 animate-pulse"></span>
//           <span className="relative z-10">Check activity</span>
//         </Button>
//       </SheetTrigger>
//       <SheetContent>
//         <SheetHeader>
//           <SheetTitle>Agent activity</SheetTitle>
//           <SheetDescription>
//             <div className={`bg-muted flex flex-col gap-5 p-2 rounded-t-5 overflow-scroll h-[800px]`}>
//               {displayedMessages.map((msg, index) => (
//                 <div className="flex gap-2" key={index}>
//                   {msg.agent === "agent" && <div className="flex items-center gap-2">
//                     <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                       <Image src={agent} alt="" className="rounded-full" />
//                     </div>
//                     <div className="">
//                       <p className="text-gray-500 text-md">Supervisor agent</p>
//                       <p className="text-gray-500 text-sm">{msg.message}</p>
//                     </div>
//                   </div>}
//                   {msg.agent === "github-agent" && <div className="flex items-center gap-2">
//                     <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                       <Image src={github} alt="" className="rounded-full" />
//                     </div>
//                     <div className="">
//                       <p className="text-gray-500 text-md">Github</p>
//                       <p className="text-gray-500 text-sm">{msg.message}</p>
//                     </div>
//                   </div>}
//                 </div>))}
//             </div>
//           </SheetDescription>
//         </SheetHeader>
//         {/* <SheetFooter>
//           <SheetClose asChild className="mt-5">
//             <Button type="submit">Close</Button>
//           </SheetClose>
//         </SheetFooter> */}
//       </SheetContent>
//     </Sheet>
//   )
// }
