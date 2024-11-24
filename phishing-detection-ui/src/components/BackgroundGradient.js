import React from "react";
import "./BackgroundGradient.css"; 

export const BackgroundGradient = ({ children }) => {
  return (
    <div className="h-screen w-screen relative overflow-hidden bg-gradient-to-br from-purple-500 via-blue-500 to-pink-500">
     
      <div className="absolute w-[500px] h-[500px] rounded-full bg-purple-400 blur-3xl animate-first"></div>

      <div className="absolute w-[400px] h-[400px] rounded-full bg-blue-300 blur-2xl animate-second"></div>
 
      <div className="absolute w-[600px] h-[600px] rounded-full bg-pink-300 blur-3xl animate-third"></div>

      <div className="absolute w-[300px] h-[300px] rounded-full bg-yellow-300 blur-2xl animate-fourth"></div>
    
      <div className="relative z-10 flex items-center justify-center h-full">
        {children}
      </div>
    </div>
  );
};
