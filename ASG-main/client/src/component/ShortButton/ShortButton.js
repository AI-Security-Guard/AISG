import { CustomButton } from './ShortButton.style.js';

function ShortButton({ txt, onClick }) {
    return (
        <>
            <CustomButton variant='contained' onClick={onClick}>
                {txt}
            </CustomButton>
        </>
    );
}

export default ShortButton;
