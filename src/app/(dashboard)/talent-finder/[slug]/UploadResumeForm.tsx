import { useState, FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';

interface ContentProps {
    setOpen: React.Dispatch<React.SetStateAction<{ open: boolean; type: string }>>;
    setResumeUpload?: React.Dispatch<React.SetStateAction<boolean>>;
}
 

export default function UploadResumeForm({ setOpen, setResumeUpload }: ContentProps) {
    const [uploadType, setUploadType] = useState('bulk');

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      setResumeUpload?.(true)
      setOpen({ open: false, type: "" })
    };

    return (
        <Card className="p-6">
        {/* <h2 className="text-xl font-bold mb-4">Setup</h2> */}
        <form onSubmit={handleSubmit}>
          {/* Job Description Upload */}
          {/* <div className="mb-4">
            <Label htmlFor="jobDescription" className="block font-medium mb-2">
              * Upload Job Description
            </Label>
            <Input id="jobDescription" name="jobDescription" type="file" className="w-full" />
          </div> */}
  
          {/* Resume Upload Type */}
          <div className="mb-8">
            <Label className="block font-medium mb-2">Resume Upload Type</Label>
            <RadioGroup value={uploadType} onValueChange={setUploadType} className="flex gap-4">
              <Label className="flex items-center gap-2">
                <RadioGroupItem value="single" /> Single
              </Label>
              <Label className="flex items-center gap-2">
                <RadioGroupItem value="bulk" /> Bulk
              </Label>
            </RadioGroup>
          </div>
  
          {/* Profile Upload */}
          <div className="mb-8">
            <Label htmlFor="profileUpload" className="block font-medium mb-2">
              * Upload Profile
            </Label>
            <Input id="profileUpload" name="profileUpload" type="file" className="w-full" />
          </div>
  
          {/* Submit Button */}
          <Button type="submit" className="w-full bg-primary text-white">Submit</Button>
        </form>
      </Card>
    )
}
