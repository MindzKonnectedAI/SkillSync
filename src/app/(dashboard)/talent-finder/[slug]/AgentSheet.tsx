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
import github from "@/image/Github.gif"

export default function AgentSheet() {

    return (
        <Sheet>
            <SheetTrigger asChild>
                <Button type="submit" size="sm" variant="outline" className="relative px-6 py-3 text-black bg-white shadow-sm rounded-full border-2 border-transparent overflow-hidden group">
                    <span className="absolute inset-0 rounded-full border-2 border-zinc-500 animate-pulse"></span>
                    <span className="relative z-10">Agents</span>
                </Button>
            </SheetTrigger>
            <SheetContent>
                <SheetHeader>
                    <SheetTitle>Worrking Agents</SheetTitle>
                    <SheetDescription>
                        <div className={`bg-muted flex flex-col gap-5 p-2 rounded-t-5`}>
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center gap-2">
                                    <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                        <Image src={agent} alt="" className="rounded-full" />
                                    </div>
                                    <div className="">
                                        <p className="text-gray-500 text-md">Supervisor agent</p>
                                        {/* <p className="text-gray-500 text-sm">{msg.message}</p> */}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                        <Image src={github} alt="" className="rounded-full" />
                                    </div>
                                    <div className="">
                                        <p className="text-gray-500 text-md">Github</p>
                                        {/* <p className="text-gray-500 text-sm">{msg.message}</p> */}
                                    </div>
                                </div>
                            </div>
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
