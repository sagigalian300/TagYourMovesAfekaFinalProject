export default function TestCard({ title, description, children, icon }) {
  return (
    <div className="bg-white flex flex-col rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="p-5 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-gray-100 rounded-md text-gray-700">
            {icon}
          </div>
          <h2 className="text-lg font-bold text-gray-900">{title}</h2>
        </div>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
      
      <div className="p-6 flex-grow flex flex-col items-center justify-center bg-white border-2 border-dashed border-gray-150 m-4 rounded-md">
        {children}
      </div>
    </div>
  );
}