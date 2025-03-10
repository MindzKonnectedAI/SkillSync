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

interface ContentProps {
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setJDContent?: React.Dispatch<React.SetStateAction<string>>;
    // setConnect: React.Dispatch<React.SetStateAction<string>>;
    // setChangeComponents: React.Dispatch<React.SetStateAction<boolean>>;
    findMatch?: () => void;

}

const ATS: React.FC<ContentProps> = ({ setOpen }: ContentProps) => {
    const connect = () => {
        // setConnect(value)
        setOpen(prev => ({ ...prev, open: false }))
    }
    return (
        <>
            <div className="flex items-center space-x-2">
                <div className="grid flex-1 gap-2">
                    <Label htmlFor="link" className="sr-only">
                        Link
                    </Label>
                    <Input
                        type="link"
                    />
                </div>
            </div>
            <DialogFooter className="sm:justify-start">
                <Button type="submit" size="sm" className="px-3" onClick={() => connect()}>
                    Connect
                </Button>
            </DialogFooter>
        </>
    )
}



const Github: React.FC<ContentProps> = ({ setOpen, findMatch, setJDContent }: ContentProps) => {
    const [file, setFile] = useState<File | null>(null);
    const [message, setMessage] = useState('');

    // const connect = () => {
    //     // setConnect(value)
    //     // setChangeComponents(true)
    //     setOpen(prev => ({ ...prev, open: false }))
    //     findMatch?.()
    // }

    interface UploadResponse {
        message: string;
    }
    const [extractedText, setExtractedText] = useState("");

    const [content, setContent] = useState('');
    const [loading, setLoading] = useState(false);

    console.log("extracted text", extractedText)
    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
        e.preventDefault();
        setJDContent?.(content)
        setOpen({ type: "", open: false })
    };

    const handleUpload = async (event: any) => {
        setFile(event)
        const file = event
        if (!file) return;

        const formData: FormData = new FormData();
        formData.append('file', file);

        try {
            setLoading(true);
            const response = await axios.post('/api/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log("response", response)
            setContent(response.data.text);
        } catch (error) {
            console.error('Upload error:', error);
            alert('Error processing PDF');
        } finally {
            setLoading(false);
        }
    }


    return (
        <form onSubmit={handleSubmit}>
            <div className="flex items-center space-x-2">
                <div className="grid flex-1 gap-2">
                    <Label htmlFor="link" className="sr-only">
                        Link
                    </Label>
                    <Input onChange={(e) => { if (e.target.files) handleUpload(e.target.files[0]); }}
                        type="file"
                    />
                    <Textarea rows={20} value={content} placeholder="Type your message here." />
                </div>
            </div>
            <DialogFooter className="sm:justify-start">
                <Button type="submit" size="sm" className="px-3 mt-5">
                    Submit
                    {/* <Copy /> */}
                </Button>
            </DialogFooter>
        </form>
    )
}

interface DialogSProps {
    open: { open: boolean; type: string };
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setJDContent: React.Dispatch<React.SetStateAction<string>>;
    // setChangeComponents: React.Dispatch<React.SetStateAction<boolean>>;
    findMatch?: () => void;
}

export default function dialog({ open, setOpen, findMatch, setJDContent }: DialogSProps) {

    return (
        <Dialog open={open.open} onOpenChange={(isOpen: boolean) => setOpen(prev => ({ ...prev, open: isOpen }))}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>
                        {open.type === "ATS" && "Connect your ATS"}
                        {open.type === "github" && "Upload job discription"}
                    </DialogTitle>
                </DialogHeader>
                {open.type === "ATS" && <ATS setOpen={setOpen} />}
                {open.type === "github" && <Github setOpen={setOpen} findMatch={findMatch} setJDContent={setJDContent} />}
            </DialogContent>
        </Dialog>
    )
}
