'use client'
import React, { useState } from 'react';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { Button } from 'antd';

export default function CollapsedButton() {
      const [collapsed, setCollapsed] = useState(false);
    
    return (
        <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{
                fontSize: '16px',
                width: 64,
                height: 64,
            }}
        />
    )
}
