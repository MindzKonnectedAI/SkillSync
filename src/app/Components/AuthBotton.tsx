'use client'
import { useState, Fragment } from "react"
import { Button } from "antd"
import AuthModal from "./AuthModal";

export default function AuthBotton() {''
    type ModaType = {
        type: string,
        visible: Boolean,
    }
    const [isVisible, setIsVisible] = useState<ModaType>({ type: "", visible: false });

    return (
        <Fragment>
            <div className="content-container">
                <div><h1>Welcome to Talent Score</h1></div>
                <div className='content-btn'>
                    <div>
                        <Button className='btn' type='primary' size='large' onClick={() => setIsVisible({ type: 'signup', visible: true })}>
                            Signup
                        </Button>
                    </div>
                    <div>
                        <Button onClick={() => setIsVisible({ type: "signin", visible: true })} className='btn' type='primary' size='large'
                        >Signin
                        </Button>
                    </div>
                </div>
            </div>
            {isVisible.visible && <AuthModal isVisible={isVisible} setIsVisible={setIsVisible} />}
        </Fragment>
    )
}
