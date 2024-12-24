"use-client"
import { Modal } from 'antd';
import Signup from './Signup';
import Signin from './Signin';

interface ModalProps {
  isVisible: { type: string; visible: boolean };
  setIsVisible: (state: { type: string; visible: boolean }) => void;
}

const AuthModal: React.FC<ModalProps> = ({ isVisible, setIsVisible }) => {
  const { type, visible } = isVisible

  return (
    <Modal title={type === "signup" && "Signup" || type === "signin" && "Signin"} open={visible}
      onOk={() => setIsVisible({ type: "", visible: false })}
      onCancel={() => setIsVisible({ type: "", visible: false })}
      footer={null}
    >
      {type === "signup" && <Signup setIsVisible={setIsVisible}/>}
      {type === "signin" && <Signin setIsVisible={setIsVisible} />}
    </Modal>
  )
}
export default AuthModal;
