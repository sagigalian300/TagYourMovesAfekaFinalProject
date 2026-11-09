export default function Button({ children, onClick, variant = 'primary', className = '', type = 'button', ...props }) {
  const baseStyle = "px-4 py-2 rounded-md text-sm font-semibold transition-colors duration-150 flex items-center justify-center w-full sm:w-auto";
  
  const variants = {
    primary: "bg-black text-white hover:bg-gray-800 border border-black",
    secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200",
    outline: "bg-transparent text-gray-900 hover:bg-gray-50 border border-gray-300"
  };

  return (
    <button 
      type={type}
      className={`${baseStyle} ${variants[variant]} ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
}