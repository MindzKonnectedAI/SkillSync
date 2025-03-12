
// import { Card, Tabs } from "antd"
// import MatchingPoint from "../../components/MatchingPoint"
// import NotMatchingPoint from "../../components/NotMatchingPoint"
// import { CheckOutlined, CloseOutlined } from '@ant-design/icons';
// import { TabSectionProps } from "@/app/types/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import {
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"

export default function TabSection() {

    return (
        <Card >
            {/* <Tabs
                defaultActiveKey="1"
                items={[
                    {
                        label: <strong className='text-[#32cd32]'>Matches</strong>,
                        key: '1',
                        children: <MatchingPoint Match={resumeAIDetails?.Match || []} />,
                        icon: <CheckOutlined style={{ fontSize: "20px", color:"#32cd32" }} />
                    },
                    {
                        label: <strong className='text-[#dc143c]'>Not a Match</strong>,
                        key: '2',
                        children: <NotMatchingPoint  Not_Match={resumeAIDetails?.Not_Match || []} />,
                        icon: <CloseOutlined style={{ fontSize: "20px", color:"#dc143c" }} />
                    },
                ]}
            /> */}
            <Tabs defaultValue="account" className="w-[400px]">
                <TabsList>
                    <TabsTrigger value="account"><strong className='text-[#32cd32]'>Matches</strong></TabsTrigger>
                    <TabsTrigger value="password"><strong className='text-[#dc143c]'>Not a Match</strong></TabsTrigger>
                </TabsList>
                <TabsContent value="account">
                    <Table>
                        {/* <TableCaption>A list of your recent invoices.</TableCaption>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-[100px]">Invoice</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Method</TableHead>
                                <TableHead className="text-right">Amount</TableHead>
                            </TableRow>
                        </TableHeader> */}
                        <TableBody>
                            <TableRow>
                                <TableCell className="font-medium">Job ID: JD001 - Matched with 85 points</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>JD002 partially matched with 72 points</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>JD003 matched with 90 points</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>JD004 partially matched with 68 points</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>JD005 matched with 88 points</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TabsContent>
                <TabsContent value="password">
                    <TableRow>
                        <TableCell className="font-medium">Job ID: JD006 - No match found</TableCell>
                    </TableRow>
                    <TableRow>
                        <TableCell>JD007 did not match any criteria</TableCell>
                    </TableRow>
                    <TableRow>
                        <TableCell>JD008 - No relevant qualifications matched</TableCell>
                    </TableRow>
                    <TableRow>
                        <TableCell>JD009 did not meet the required points</TableCell>
                    </TableRow>
                    <TableRow>
                        <TableCell>JD010 - No alignment with job requirements</TableCell>
                    </TableRow>

                </TabsContent>
            </Tabs>
        </Card>
    )
}
