import { CustomButton } from "./LongButton.style.js";

function LongButton({ txt, onClick }) {
    return (
        <>
            <CustomButton variant="contained" onClick={onClick}>
                {txt}
            </CustomButton>
        </>
    );
}

export default LongButton;
