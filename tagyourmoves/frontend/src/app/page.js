import Link from 'next/link';
import Button from '@/components/ui/Button';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-8">
      <h1 className="text-5xl font-extrabold text-blue-600 mb-6 tracking-tight">
        Tag Your Moves
      </h1>
      <p className="text-xl text-gray-700 font-medium mb-2">
        System Fullstack - Homepage
      </p>
      <p className="text-gray-500 mb-8 text-center max-w-md">
        (The full video upload and tagging interface will be implemented here later)
      </p>
      
      <Link href="/tests">
        <Button variant="outline" className="px-6 py-3">
          Go to Tests Dashboard &rarr;
        </Button>
      </Link>
    </main>
  );
}