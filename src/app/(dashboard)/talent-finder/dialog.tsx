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

interface ContentProps {
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
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



const Github: React.FC<ContentProps> = ({ setOpen, findMatch }: ContentProps) => {
    const connect = () => {
        // setConnect(value)
        // setChangeComponents(true)
        setOpen(prev => ({ ...prev, open: false }))
        findMatch?.()
    }
    return (
        <>
            <div className="flex items-center space-x-2">
                <div className="grid flex-1 gap-2">
                    <Label htmlFor="link" className="sr-only">
                        Link
                    </Label>
                    <Input
                        type="file"
                    />
                </div>
            </div>
            <DialogFooter className="sm:justify-start">
                <Button type="submit" size="sm" className="px-3" onClick={() => connect()}>
                    Submit
                    {/* <Copy /> */}
                </Button>
            </DialogFooter>
        </>
    )
}

interface DialogSProps {
    open: { open: boolean; type: string };
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    // setChangeComponents: React.Dispatch<React.SetStateAction<boolean>>;
    findMatch?: () => void;
}

export default function dialog({ open, setOpen, findMatch }: DialogSProps) {

    return (
        <Dialog open={open.open} onOpenChange={(isOpen: boolean) => setOpen(prev => ({ ...prev, open: isOpen }))}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>
                        {open.type === "ATS" && "Connect your ATS"}
                        {open.type === "github" && "Upload job discription"}
                    </DialogTitle>
                </DialogHeader>
                {open.type === "ATS" && <ATS setOpen={setOpen}  />}
                {open.type === "github" && <Github setOpen={setOpen} findMatch={findMatch} />}
            </DialogContent>
        </Dialog>
    )
}
