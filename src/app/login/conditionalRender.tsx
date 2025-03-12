"use client"
import { useState } from 'react'
import AgentSelction from './agentSelction'
import { LoginForm } from "@/components/login-form"

export default function ConditionalRender() {
    const [changeComponent, setComponent] = useState(true)
  return (
    <div>{changeComponent ? <LoginForm setComponent={setComponent}/> : <AgentSelction/>}</div>
  )
}
