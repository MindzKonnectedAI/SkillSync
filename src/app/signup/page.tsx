import Signup from "./Signup"
import Navber from "@/Layout/Navber";

export default function page() {
    return (
        <>
            <Navber />
            <div className="flex flex-col items-center justify-center gap-6 p-6 md:p-4 bg-muted h-[90dvh]">
                <div className="flex w-full max-w-[35%] flex-col gap-6 shadow-md">
                    <Signup />
                </div>
            </div>
        </>
    )
}
