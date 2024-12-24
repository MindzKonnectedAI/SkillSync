import React, { Fragment, useState } from 'react';
import type { FormProps } from 'antd';
import { Button, Form, Input, notification } from 'antd';
import type { NotificationArgsProps } from 'antd';
import { postRequest } from '@/utils/requests';

type FieldType = {
    firstname?: string;
    lastname?: string;
    email?: string;
    password?: string;
};

export default function Signin({ setIsVisible }) {
    const [isLoading, setIsLoading] = useState(false)
    const [api, contextHolder] = notification.useNotification()

    const openNotificationWithIcon = (type, message) => {
        api[type]({
          message: message,
        //   description:
        //     'This is the content of the notification. This is the content of the notification. This is the content of the notification.',
        });
      };

    const signup = async (value: FieldType) => {
        console.log("Signup", value)
        try {
            setIsLoading(true)
            const res = await postRequest(`auth/signup`, value)
            console.log("res: ", res)
            openNotificationWithIcon("success", res.message)
            // setIsVisible(false)
        } catch (error) {

        }
        setIsLoading(false)
    }

    const onFinish: FormProps<FieldType>['onFinish'] = (values) => {
        console.log('Success:', values);
        signup(values);
    };

    const onFinishFailed: FormProps<FieldType>['onFinishFailed'] = (errorInfo) => {
        console.log('Failed:', errorInfo);
    };

    return (
        <Fragment>

            {contextHolder}

            <Form
                layout='vertical'
                name="basic"
                onFinish={onFinish}
                onFinishFailed={onFinishFailed}
                autoComplete="off"
            >
                <Form.Item<FieldType>
                    label="First name"
                    name="firstname"
                    rules={[{ required: true, message: 'Please input your first name!' }]}
                >
                    <Input />
                </Form.Item>
                <Form.Item<FieldType>
                    label="Last name"
                    name="lastname"
                    rules={[{ required: true, message: 'Please input your last name!' }]}
                >
                    <Input />
                </Form.Item>
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
        </Fragment>
    )
}

