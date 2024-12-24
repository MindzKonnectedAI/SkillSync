'use client'
import { useState, useEffect } from 'react';
import { getRequest } from '@/utils/requests';
import { Row, Col, Popover, Avatar, Button } from 'antd';
import { UserOutlined } from '@ant-design/icons';
import { useRouter } from 'next/navigation'

export default function UserDetails() {

  const [isLoading, setIsLoading] = useState(false)
  const [details, setUserDetails] = useState({})
  const router = useRouter()

  const userProfile = async () => {
    try {
      setIsLoading(true)
      const res = await getRequest("auth/get-user-details")
      console.log(res)
      setUserDetails(res?.data)
    } catch (error) {

    }
    setIsLoading(false)
  }

  useEffect(() => {
    userProfile()
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('auth')
    router.push("/")  // replace with your logout route path
  }

  const content = (
    <div>
      <Button onClick={handleLogout}>Logout</Button>
    </div>
  );

  return (
    <Row gutter={15} align="middle">
      <Col><strong>{details?.email}</strong></Col>
      <Col>
        <Popover content={content} title="">
          <Avatar size={34} icon={<UserOutlined />} />
        </Popover>
      </Col>
    </Row>
  )
}
