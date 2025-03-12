'use client'
import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { useState } from 'react'
import { Textarea } from "@/components/ui/textarea"
import axios from 'axios'
import UploadResumeForm from "./UploadResumeForm"

interface DialogSProps {
    open: { open: boolean; type: string };
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setResumeUpload: React.Dispatch<React.SetStateAction<boolean>>;
}

export default function dialog({ open, setOpen, setResumeUpload }: DialogSProps) {

    return (
        <Dialog open={open.open} onOpenChange={(isOpen: boolean) => setOpen(prev => ({ ...prev, open: isOpen }))}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>
                        {open.type === "resume" && "Upload resume"}
                    </DialogTitle>
                </DialogHeader>
                {open.type === "resume" && <UploadResumeForm setOpen={setOpen} setResumeUpload={setResumeUpload}/>}
            </DialogContent>
        </Dialog>
    )
}
