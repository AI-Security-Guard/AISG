import { CustomInput } from "./Input.style.js";

function Input({ txt, ...props }) {
    return <CustomInput label={txt} variant="outlined" {...props} />;
}

export default Input;
