import React from 'react'
import { motion, HTMLMotionProps } from 'framer-motion'
import { cn } from '../../utils/cn'

interface AnimatedCardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode
  className?: string
  delay?: number
  duration?: number
  hover?: boolean
  tap?: boolean
}

const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  className,
  delay = 0,
  duration = 0.3,
  hover = true,
  tap = true,
  ...props
}) => {
  const variants = {
    hidden: {
      opacity: 0,
      y: 20,
      scale: 0.95,
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration,
        delay,
        ease: [0.25, 0.46, 0.45, 0.94],
      },
    },
  }

  const hoverVariants = hover ? {
    scale: 1.02,
    y: -4,
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    transition: {
      duration: 0.2,
      ease: 'easeOut',
    },
  } : {}

  const tapVariants = tap ? {
    scale: 0.98,
    transition: {
      duration: 0.1,
    },
  } : {}

  return (
    <motion.div
      className={cn(
        'bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700',
        className
      )}
      variants={variants}
      initial="hidden"
      animate="visible"
      whileHover={hoverVariants}
      whileTap={tapVariants}
      {...props}
    >
      {children}
    </motion.div>
  )
}

export default AnimatedCard