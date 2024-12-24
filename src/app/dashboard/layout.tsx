"use client"
import Image from 'next/image';
import logo from '../../images/DataCurvelogo.png';
import UserDetails from './components/UserDetails';
import { useRouter } from 'next/navigation'

export default function dashboardLayout({
    children,
}: {
    children: React.ReactNode
}) {

    const router = useRouter()
    if(!localStorage.getItem("auth")) {
        return router.push("/")
      }

    return (
        <div className="layout-container">
            <div className="header-container">
                <div className='img-container'>
                    <Image src={logo} width={500} height={300} alt="" />
                </div>
                <div>
                    <UserDetails />
                </div>
            </div>
            <div className="content-container-common">{children}</div>
        </div>
    )
}