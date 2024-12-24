"use-client"
import { Modal } from 'antd';
import Signup from './Signup';
import Signin from './Signin';
import Setup from './Setup';

interface ModalProps {
  isVisible: { type: string; visible: boolean };
  setIsVisible: (state: { type: string; visible: boolean }) => void;
}

const SetupModal: React.FC<ModalProps> = ({ isVisible, setIsVisible, getJDDetails }) => {
  const { type, visible } = isVisible

  return (
    <Modal title={type === "setup" && "Setup"} open={visible}
      onOk={() => setIsVisible({ type: "", visible: false })}
      onCancel={() => setIsVisible({ type: "", visible: false })}
      footer={null}
    >
      {type === "setup" && <Setup setIsVisible={setIsVisible} getJDDetails={getJDDetails} />}
    </Modal>
  )
}
export default SetupModal;
