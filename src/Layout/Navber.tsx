import Image from "next/image"

export default function Navber() {
    return (
        <div className="border border-b-2 px-4 py-2">
            <div>
                <Image src="/DataCurvelogo.png"  alt="logo" width={150} height={100} />
            </div>
        </div>
    )
}
