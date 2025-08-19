export { default as AnimatedCard } from './AnimatedCard'
export { default as AnimatedList } from './AnimatedList'
export { default as PageTransition } from './PageTransition'
export { default as LoadingSpinner, LoadingOverlay } from './LoadingSpinner'
export { default as AnimatedModal } from './AnimatedModal'

// Animation utilities and variants
export const fadeInUp = {
  hidden: {
    opacity: 0,
    y: 20,
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const fadeInDown = {
  hidden: {
    opacity: 0,
    y: -20,
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const fadeInLeft = {
  hidden: {
    opacity: 0,
    x: -20,
  },
  visible: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const fadeInRight = {
  hidden: {
    opacity: 0,
    x: 20,
  },
  visible: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const scaleIn = {
  hidden: {
    opacity: 0,
    scale: 0.8,
  },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const slideInUp = {
  hidden: {
    y: '100%',
    opacity: 0,
  },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export const staggerContainer = {
  hidden: {
    opacity: 0,
  },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
}

export const hoverScale = {
  scale: 1.05,
  transition: {
    duration: 0.2,
    ease: 'easeOut',
  },
}

export const tapScale = {
  scale: 0.95,
  transition: {
    duration: 0.1,
  },
}

// Common easing functions
export const easing = {
  easeInOut: [0.25, 0.46, 0.45, 0.94],
  easeOut: [0.25, 0.46, 0.45, 0.94],
  easeIn: [0.42, 0, 1, 1],
  bounce: [0.68, -0.55, 0.265, 1.55],
}

// Duration constants
export const duration = {
  fast: 0.2,
  normal: 0.3,
  slow: 0.5,
}