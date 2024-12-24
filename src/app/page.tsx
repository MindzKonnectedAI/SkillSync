'use client'
import Image from 'next/image';
import logo from '../images/DataCurvelogo.png';
// import { Button } from "antd"
// import bg from "../images/DataCurvelogo.png"
// import { Fragment } from 'react';
import AuthBotton from './Components/AuthBotton';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter()
  if(localStorage.getItem("auth")) {
    return router.push("/dashboard")
  }

  return (
    <div className="layout-container">
      <div className="header-container">
        <div className='img-container'>
          <Image src={logo} width={500} height={300} alt="" />
        </div>
      </div>
      <div className='main-container'>
        <div className="content-container-image" />
        <AuthBotton />
        {/* <div className="content-container">
          <div><h1>Welcome to Talent Score</h1></div>
          <div className='content-btn'>
            <div><Button className='btn' type='primary' size='large' onClick={() => setIsVisible({ type: '', visible: true })}>Signup</Button></div>
            <div><Button className='btn' type='primary' size='large'>Signin</Button></div>
          </div>
        </div> */}
      </div>
    </div>
  );
}
