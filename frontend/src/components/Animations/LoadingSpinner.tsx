import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../utils/cn'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  color?: 'primary' | 'secondary' | 'white'
  className?: string
  text?: string
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'primary',
  className,
  text,
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  }

  const colorClasses = {
    primary: 'border-blue-600',
    secondary: 'border-gray-600',
    white: 'border-white',
  }

  const spinVariants = {
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: 'linear',
      },
    },
  }

  const pulseVariants = {
    animate: {
      scale: [1, 1.1, 1],
      opacity: [0.7, 1, 0.7],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  }

  return (
    <div className={cn('flex flex-col items-center justify-center space-y-2', className)}>
      <div className="relative">
        {/* Outer ring */}
        <motion.div
          className={cn(
            'border-2 border-transparent rounded-full',
            sizeClasses[size],
            `border-t-2 ${colorClasses[color]}`
          )}
          variants={spinVariants}
          animate="animate"
        />
        
        {/* Inner pulse */}
        <motion.div
          className={cn(
            'absolute inset-1 rounded-full',
            colorClasses[color].replace('border-', 'bg-'),
            'opacity-20'
          )}
          variants={pulseVariants}
          animate="animate"
        />
      </div>
      
      {text && (
        <motion.p
          className="text-sm text-gray-600 dark:text-gray-400 font-medium"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )
}

// Full page loading overlay
export const LoadingOverlay: React.FC<{ isVisible: boolean; text?: string }> = ({
  isVisible,
  text = 'Carregando...',
}) => {
  if (!isVisible) return null

  return (
    <motion.div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <motion.div
        className="bg-white dark:bg-gray-800 rounded-lg p-8 shadow-xl"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.8, opacity: 0 }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        <LoadingSpinner size="lg" text={text} />
      </motion.div>
    </motion.div>
  )
}

export default LoadingSpinner