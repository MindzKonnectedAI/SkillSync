import Header from './Header';
import './Layout.css';
import { ReactNode } from 'react'

export default function ContainerLayout(children: ReactNode) {
    return (
        <div className="layout-container">
            <Header />
            <div className="content-container">{children}</div>
        </div>
    );
}
