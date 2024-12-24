import React, { useState, useEffect } from 'react';
import type { FormProps } from 'antd';
import { Button, Form, Input, notification, Upload, Radio } from 'antd';
import axios from 'axios';
import type { NotificationArgsProps } from 'antd';
import { getRequest, postRequest1 } from '@/utils/requests';
import { ConsoleSqlOutlined, InboxOutlined, UploadOutlined } from '@ant-design/icons';

type FieldType = {
    firstname?: string;
    lastname?: string;
    email?: string;
    password?: string;
};
type NotificationPlacement = NotificationArgsProps['placement'];

const openNotification = (placement: NotificationPlacement) => {
    notification.success({
        message: `Signup successfully`,
        //   description: "Signup successfully",
        placement,
    });
};

const normFile = (e: any) => {
    console.log('Upload event:', e);
    if (Array.isArray(e)) {
        return e;
    }
    return e?.fileList;
};

const normFileMulti = (e: any) => {
    console.log('Upload event:', e);
    if (Array.isArray(e)) {
        return e;
    }
    return e?.fileList;
};

export default function Setup({ setIsVisible, getJDDetails }) {
    const [isLoading, setIsLoading] = useState(false)
    const [isGetJSLoading, setGetJDIsLoading] = useState(false)
    const [jdDettails, setjdDettails] = useState()
    const [uploadResume, setUploadResume] = useState(true)
    const [api, contextHolder] = notification.useNotification()


    // const getJDDetails = async () => {
    //     try {
    //         setGetJDIsLoading(true)
    //       const res = await getRequest("auth/get-job-description")
    //     //   console.log(res)
    //       setjdDettails(res?.data)
    //     } catch (error) {

    //     }
    //     setGetJDIsLoading(false)
    //   }

    //   useEffect(() => {
    //     getJDDetails()
    //   }, [])

    const openNotificationWithIcon = (type, message) => {
        api[type]({
            message: message,
            //   description:
            //     'This is the content of the notification. This is the content of the notification. This is the content of the notification.',
        });
    };

    // const uploadJD = async (value: FieldType) => {
    //     console.log("Signup", value)
    //     let formData = new FormData();    //formdata object
    //     formData.append('file', value?.file[0]?.originFileObj);

    //     try {
    //         setIsLoading(true)
    //         const res = await postRequest1(`auth/upload-job-description`, formData)
    //         console.log("res: " + res)
    //         openNotificationWithIcon("success", res.message)
    //         setIsVisible(false)
    //     } catch (error) {

    //     }
    //     setIsLoading(false)
    // }

    const uploadResumeInBulk = async (value: FieldType) => {
        console.log("value", value)
        let formData = new FormData();    //formdata object
        formData.append('file', value?.file[0]?.originFileObj);

        try {
            setIsLoading(true)
            const JDres = await postRequest1(`auth/upload-job-description`, formData)
            console.log("JDres: ", JDres)
            let resumeFormData = new FormData();

            const files = value.fileList?.map(file => file.originFileObj)
            console.log("files: ", files)
            // resumeFormData.append('files', files);
            files.forEach((file) => {
                resumeFormData.append('files', file);
            })
            resumeFormData.append('job_description_id', JDres?._id);

            const res = await postRequest1(`auth/upload-resume`, resumeFormData)
            console.log("res: " + res)
            openNotificationWithIcon("success", res.message)
            getJDDetails()
            setIsVisible(false)
        } catch (error) {

        }
        setIsLoading(false)
    }

    const onFinish: FormProps<FieldType>['onFinish'] = (values) => {
        console.log('Success:', values);

        uploadResumeInBulk(values)
        // uploadJD(values);
    };

    const onFinishFailed: FormProps<FieldType>['onFinishFailed'] = (errorInfo) => {
        console.log('Failed:', errorInfo);
    };

    return (
        <Form
            layout='vertical'
            name="basic"
            onFinish={onFinish}
            onFinishFailed={onFinishFailed}
            autoComplete="off"
        >
            <Form.Item
                name="file"
                label="Upload job discription"
                valuePropName="fileList"
                getValueFromEvent={normFile}
            // extra="longgggggggggggggggggggggggggggggggggg"
            >

                <Upload maxCount={1} name="logo" action="none" listType="picture">
                    <Button icon={<UploadOutlined />}>Click to upload job discription</Button>
                </Upload>
            </Form.Item>
            <Form.Item label="Resume upload type">
                <Radio.Group defaultValue={"bulk"} onChange={(e) => setUploadResume(e.target.value === "bulk" ? true : false)}>
                    <Radio value="single"> Single </Radio>
                    <Radio value="bulk"> Bulk </Radio>
                </Radio.Group>
            </Form.Item>
            <Form.Item
                name="fileList"
                label="Upload resume"
                valuePropName="fileList"
                getValueFromEvent={normFileMulti}
            // extra="longgggggggggggggggggggggggggggggggggg"
            >
                <Upload name="logo" action={false} listType="picture" directory={uploadResume ? true : false}>
                    <Button icon={<UploadOutlined />}>Click to upload resume</Button>
                </Upload>
            </Form.Item>
            <Form.Item label={null}>
                <Button loading={isLoading} type="primary" htmlType="submit">
                    Submit
                </Button>
            </Form.Item>
        </Form>
    )
}

