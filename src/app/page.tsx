import { LoginForm } from "@/components/login-form";
import Navber from "@/Layout/Navber";
import ConditionalRender from "./login/conditionalRender";

export default function Home() {
  return (
    <>
      <Navber />
      <div className="flex flex-col items-center justify-center gap-6 p-6 md:p-4 bg-muted h-[90dvh]">
        <div className="flex w-full max-w-[35%] flex-col gap-6 shadow-md">
          {/* <LoginForm /> */}
          <ConditionalRender />
        </div>
      </div>
    </>
  );
}
