// import { Progress } from "antd"
// import TabSection from "./components/TabSection";
import alex from "@/image/alex.jpg"
import Image from "next/image";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import TabSection from "./TabSection";

export default function ContentSections() {

    const resumeAIDetails = {
        Name: 'John Doe',
        Email: 'john@gmail.com',
        Profile_Match: 'Good',
        Score: 88,
    };

    return (
        <div className="w-full">
            <div className='bg-[#fff] p-3 '>
                {/* <div className='h-[90px] rounded-md bg-[#49a2acd9]'></div> */}
                <div className='h-[90px] rounded-md bg-[#F5F5F5]'></div>
                {/* <div className='h-[90px] rounded-md bg-cyan-100'></div> */}
                <div className='flex justify-center items-center flex-col relative -top-5 h-[100px]'>
                    <div className='bg-slate-200 rounded-full p-2 shadow-sm'>
                        {/* <Avatar
                            size={{ xs: 24, sm: 32, md: 40, lg: 64, xl: 70, xxl: 100 }}
                        // src={alex}
                        >
                        </Avatar> */}
                        <Avatar>
                            <Image src={alex} alt='Alex' width={100} height={100} className='rounded-full' />
                            {/* <AvatarImage  src={alex} /> */}
                            {/* <AvatarFallback>CN</AvatarFallback> */}
                        </Avatar>
                    </div>
                    <div className='text-center'>
                        <div className='mb-2'>
                            <strong className='text-gray-700 text-xl'>{resumeAIDetails?.Name}</strong>
                        </div>
                        <div>
                            <p className='text-gray-700 font-medium'>{resumeAIDetails?.Email}</p>
                        </div>
                    </div>
                </div>
            </div>
            <div className='py-3'>
                <div className="flex  gap-5">
                    <div className="basis-1/2 ">
                        <Card className="p-5 h-[100px]">
                            <div className='flex justify-between items-center'>
                                <div className='flex items-center gap-3 h-[65px]'>
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="44" height="44" color={resumeAIDetails?.Profile_Match === 'Bad' && "#dc143c" || resumeAIDetails?.Profile_Match === 'Average' && "#FCC737" || resumeAIDetails?.Profile_Match === 'Good' && "#32cd32" || "#000000"} fill="none">
                                        <path d="M2.5 12C2.5 7.52166 2.5 5.28249 3.89124 3.89124C5.28249 2.5 7.52166 2.5 12 2.5C16.4783 2.5 18.7175 2.5 20.1088 3.89124C21.5 5.28249 21.5 7.52166 21.5 12C21.5 16.4783 21.5 18.7175 20.1088 20.1088C18.7175 21.5 16.4783 21.5 12 21.5C7.52166 21.5 5.28249 21.5 3.89124 20.1088C2.5 18.7175 2.5 16.4783 2.5 12Z" stroke="currentColor" strokeWidth="2.5" />
                                        <path d="M7.5 17C9.8317 14.5578 14.1432 14.4428 16.5 17M14.4951 9.5C14.4951 10.8807 13.3742 12 11.9915 12C10.6089 12 9.48797 10.8807 9.48797 9.5C9.48797 8.11929 10.6089 7 11.9915 7C13.3742 7 14.4951 8.11929 14.4951 9.5Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                                    </svg>
                                    <strong>Profile Match</strong>
                                </div>
                                <div><strong className={`font-bold text-xl ${resumeAIDetails?.Profile_Match === 'Bad' && "text-[#dc143c]" || resumeAIDetails?.Profile_Match === 'Average' && "text-yellow-500" || resumeAIDetails?.Profile_Match === 'Good' && "text-[#32cd32]"}`}>{resumeAIDetails?.Profile_Match}</strong></div>
                            </div>
                        </Card>
                    </div>
                    <div className="basis-1/2">
                        <Card className="p-5  h-[100px]">
                            <div className='flex justify-between items-center'>
                                <div className='flex items-center gap-3'>
                                    <div>
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="44" height="44" color={resumeAIDetails?.Profile_Match === 'Bad' && "#dc143c" || resumeAIDetails?.Profile_Match === 'Average' && "#FCC737" || resumeAIDetails?.Profile_Match === 'Good' && "#32cd32" || "#000000"} fill="none">
                                            <path d="M11.0809 13.152L8 7L13.4196 11.2796C14.1901 11.888 14.1941 13.0472 13.4277 13.6607C12.6614 14.2743 11.5189 14.0266 11.0809 13.152Z" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                                            <path d="M5 4.82C3.14864 6.63902 2 9.17385 2 11.9776C2 17.5129 6.47715 22.0001 12 22.0001C17.5228 22.0001 22 17.5129 22 11.9776C22 7.1242 18.5581 3.07656 13.9872 2.15288C13.1512 1.98394 12.7332 1.89947 12.3666 2.20022C12 2.50097 12 2.98714 12 3.95949V4.96175" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </div>
                                    <strong className='text-xl'>Score</strong>
                                </div>
                                <div>
                                    <Progress
                                        // strokeColor={resumeAIDetails?.Profile_Match === 'Bad' && "#dc143c" || resumeAIDetails?.Profile_Match === 'Average' && "#FCC737" || resumeAIDetails?.Profile_Match === 'Good' && "#32cd32" || "#000000"}
                                        value={88} />
                                </div>
                            </div>
                        </Card>
                    </div>
                </div>
            </div>
            <TabSection />
        </div>
    )
}
