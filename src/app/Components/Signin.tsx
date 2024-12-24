// 'use client'
import React, { useState } from 'react';
import type { FormProps } from 'antd';
import { Button, Form, Input, notification } from 'antd';
import type { NotificationArgsProps } from 'antd';
import { postRequest } from '@/utils/requests';
import { useRouter } from 'next/navigation'

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

export default function Signup({ setIsVisible }) {
    const router = useRouter()

    const [isLoading, setIsLoading] = useState(false)
    

    const signin = async(value: FieldType) => {
        console.log("Signup", value)
        try {
            setIsLoading(true)
            const res = await postRequest(`auth/login`, value)
            console.log("res: ", res)
            localStorage.setItem("auth", res.token)

            router.push("/dashboard")
            setIsVisible(false)
            openNotification("topRight")
        } catch (error) {
            
        }
        setIsLoading(false)
        
    }
    
    const onFinish: FormProps<FieldType>['onFinish'] = (values) => {
        console.log('Success:', values);
        signin(values);
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
                name={'email'}
                label="Email"
                rules={[
                    {
                        type: 'email',
                        required: true, message: 'Please input your last name!'
                    },
                ]}
            >
                <Input />
            </Form.Item>
            <Form.Item<FieldType>
                label="Password"
                name="password"
                rules={[{ required: true, message: 'Please input your password!' }]}
            >
                <Input.Password />
            </Form.Item>
            <Form.Item label={null}>
                <Button loading={isLoading} type="primary" htmlType="submit">
                    Submit
                </Button>
            </Form.Item>
        </Form>
    )
}

