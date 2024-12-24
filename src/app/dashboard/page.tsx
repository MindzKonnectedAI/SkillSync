'use client'

import React, { useState, useEffect } from 'react';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UploadOutlined,
  UserOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons';
import { Button, Layout, Menu, theme, Row, Col, Spin, Skeleton } from 'antd';
import SetupModal from './components/SetupModal';
import { getRequest } from '@/utils/requests';
const { Header, Sider, Content } = Layout;
import { useRouter } from 'next/navigation';
import { useSearchParams } from 'next/navigation';

export default function Page() {
  type ModaType = {
    type: string,
    visible: Boolean,
  }
  const [isVisible, setIsVisible] = useState<ModaType>({ type: "", visible: false });
  const [collapsed, setCollapsed] = useState(false);
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const [isLoading, setIsLoading] = useState(false)
  const [isReumseLoading, setResumeIsLoading] = useState(false)
  const [jdDettails, setjdDettails] = useState()
  const [reumseDettails, setReumseDettails] = useState()
  const router = useRouter()
  const searchParams = useSearchParams()

  const params = searchParams.get('query')
  const rsquery = searchParams.get('rsquery')

  // console.log("params: ", params)
  console.log("rsquery: ", rsquery)

  const getJDDetails = async () => {
    try {
      setIsLoading(true)
      const res = await getRequest("auth/get-job-description")
      console.log(res)
      setjdDettails(res?.data)
      router.push(`dashboard/?query=${res?.data[0]?.id}`)
      // router.push({
      //   pathname: '/about',
      //   query: { name: res?.data[0]?.id }
      // })
      // `/${res?.data[0]?.id}`})
      getResumeDetails(res?.data[0]?.id)
    } catch (error) {

    }
    setIsLoading(false)
  }

  const getResumeDetails = async (job_description_id) => {
    try {
      setResumeIsLoading(true)
      const res = await getRequest(`auth/get-resume?job_description_id=${job_description_id}`)
      console.log(res)
      setReumseDettails(res?.data)
      router.push(`dashboard/?query=${job_description_id}&rsquery=${res?.data[0]?.id}`)

    } catch (error) {

    }
    setResumeIsLoading(false)
  }

  useEffect(() => {
    getJDDetails()
  }, [])

  const contentStyle = {
    textAlign: 'center',
    minHeight: 120,
    lineHeight: '120px',
    color: '#fff',
    backgroundColor: '#0958d9',
  };
  const siderStyle = {
    textAlign: 'center',
    lineHeight: '120px',
    color: '#fff',
    backgroundColor: '#fff',
  };

  return (
    <Layout style={{ height: "100%" }} >
      <Sider width="16%" trigger={null} collapsible collapsed={collapsed}>
        <div className="demo-logo-vertical" />
        <Spin spinning={isLoading} style={{ height: "300px" }}>
          <Menu
            theme="dark"
            mode="inline"
            defaultSelectedKeys={[params]}
            selectedKeys={[params]}
            items={jdDettails?.map((icon) => ({
              key: icon.id,
              // icon: React.createElement(icon?.icon),
              label: (
                <div onClick={() => getResumeDetails(icon.id)}>
                  {icon.filename}
                </div>
              ),
            }))}
          />
        </Spin>
      </Sider>
      <Layout>
        <Header style={{ padding: 0, background: colorBgContainer }}>
          <Row gutter={5} justify="space-between" align="middle">
            <Col>
              {/* <Button
                type="text"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed(!collapsed)}
                style={{
                  fontSize: '16px',
                  width: 64,
                  height: 64,
                }}
              /> */}
            </Col>
            <Col>
              <Button
                type='primary'
                onClick={() => setIsVisible({ type: "setup", visible: true })}
                style={{
                  fontSize: '16px',
                  marginRight: '20px',
                }}
              >
                Setup
              </Button>
            </Col>
          </Row>
        </Header>
        {/* <Content
          style={{
            margin: '24px 16px',
            padding: 24,
            minHeight: 280,
            // background: colorBgContainer,
            borderRadius: borderRadiusLG,
          }}
        >
          Content
        </Content> */}
        <Layout>
          <Content >Content</Content>
          <Sider width="20%" style={siderStyle}>
            <Spin spinning={isLoading || isReumseLoading} style={{ height: "300px" }}>
              <Menu
                theme="light"
                mode="inline"
                defaultSelectedKeys={[rsquery]}
                selectedKeys={[rsquery]}
                items={reumseDettails?.map((icon) => ({
                  key: icon.id,
                  // icon: React.createElement(icon?.icon),
                  label: (
                    <div>{console.log(icon)}
                      {icon.filename}
                    </div>
                  ),
                }))}
              />
            </Spin>
          </Sider>
        </Layout>
        {isVisible.visible && <SetupModal getJDDetails={getJDDetails} isVisible={isVisible} setIsVisible={setIsVisible} />}
      </Layout>
      {/* <Sider trigger={null} collapsible collapsed={collapsed} style={{ background: "#fff" }}>
        <div className="demo-logo-vertical" />
        <Spin spinning={isLoading} style={{ height: "300px" }}>

          <Menu
            theme="light"
            mode="inline"
            defaultSelectedKeys={['1']}
            items={reumseDettails?.map((icon) => ({
              key: icon.user_id,
              // icon: React.createElement(icon?.icon),
              label: (
                <div>
                  {icon.filename}
                </div>
              ),
            }))}
          />
        </Spin>
      </Sider> */}
    </Layout>
  )
}
