import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Share2, Copy, Twitter, Facebook, WhatsApp, Link, Check } from 'lucide-react'
import { usePWA } from '../../hooks/usePWA'

interface ShareButtonProps {
  title?: string
  text?: string
  url?: string
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  className?: string
}

const ShareButton: React.FC<ShareButtonProps> = ({
  title = 'Football Fouls Analytics',
  text = 'Confira esta análise de faltas no futebol!',
  url = window.location.href,
  variant = 'primary',
  size = 'md',
  showLabel = true,
  className = ''
}) => {
  const { shareContent } = usePWA()
  const [isSharing, setIsSharing] = useState(false)
  const [showFallback, setShowFallback] = useState(false)
  const [copied, setCopied] = useState(false)

  const variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white',
    ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
  }

  const sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  }

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  }

  const handleShare = async () => {
    setIsSharing(true)
    
    try {
      await shareContent({ title, text, url })
    } catch (error) {
      // If native sharing fails, show fallback options
      setShowFallback(true)
    } finally {
      setIsSharing(false)
    }
  }

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy link:', error)
    }
  }

  const shareToSocial = (platform: string) => {
    const encodedUrl = encodeURIComponent(url)
    const encodedText = encodeURIComponent(`${title} - ${text}`)
    
    const urls = {
      twitter: `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}`,
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`,
      whatsapp: `https://wa.me/?text=${encodedText}%20${encodedUrl}`
    }
    
    window.open(urls[platform as keyof typeof urls], '_blank', 'width=600,height=400')
    setShowFallback(false)
  }

  return (
    <>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleShare}
        disabled={isSharing}
        className={`
          inline-flex items-center space-x-2 rounded-lg font-medium transition-all
          ${variantClasses[variant]} ${sizeClasses[size]} ${className}
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
      >
        <motion.div
          animate={isSharing ? { rotate: 360 } : {}}
          transition={{ duration: 1, repeat: isSharing ? Infinity : 0, ease: "linear" }}
        >
          <Share2 className={iconSizes[size]} />
        </motion.div>
        {showLabel && (
          <span>{isSharing ? 'Compartilhando...' : 'Compartilhar'}</span>
        )}
      </motion.button>

      {/* Fallback Share Modal */}
      {showFallback && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 w-full max-w-sm"
          >
            <div className="text-center mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Compartilhar
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Escolha como deseja compartilhar
              </p>
            </div>

            <div className="space-y-3">
              {/* Copy Link */}
              <button
                onClick={handleCopyLink}
                className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                  {copied ? (
                    <Check className="w-5 h-5 text-green-600" />
                  ) : (
                    <Copy className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                  )}
                </div>
                <div className="flex-1 text-left">
                  <p className="font-medium text-gray-900 dark:text-white">
                    {copied ? 'Link copiado!' : 'Copiar link'}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Copiar para área de transferência
                  </p>
                </div>
              </button>

              {/* WhatsApp */}
              <button
                onClick={() => shareToSocial('whatsapp')}
                className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
                  <WhatsApp className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <div className="flex-1 text-left">
                  <p className="font-medium text-gray-900 dark:text-white">
                    WhatsApp
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Compartilhar via WhatsApp
                  </p>
                </div>
              </button>

              {/* Twitter */}
              <button
                onClick={() => shareToSocial('twitter')}
                className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <Twitter className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1 text-left">
                  <p className="font-medium text-gray-900 dark:text-white">
                    Twitter
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Compartilhar no Twitter
                  </p>
                </div>
              </button>

              {/* Facebook */}
              <button
                onClick={() => shareToSocial('facebook')}
                className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <Facebook className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1 text-left">
                  <p className="font-medium text-gray-900 dark:text-white">
                    Facebook
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Compartilhar no Facebook
                  </p>
                </div>
              </button>
            </div>

            <button
              onClick={() => setShowFallback(false)}
              className="w-full mt-6 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Cancelar
            </button>
          </motion.div>
        </div>
      )}
    </>
  )
}

// Quick Share Component for specific content
interface QuickShareProps {
  data: {
    title: string
    description: string
    url?: string
    image?: string
  }
  className?: string
}

export const QuickShare: React.FC<QuickShareProps> = ({ data, className = '' }) => {
  const { shareContent } = usePWA()
  const [isSharing, setIsSharing] = useState(false)

  const handleQuickShare = async () => {
    setIsSharing(true)
    
    try {
      await shareContent({
        title: data.title,
        text: data.description,
        url: data.url || window.location.href
      })
    } catch (error) {
      console.error('Quick share failed:', error)
    } finally {
      setIsSharing(false)
    }
  }

  return (
    <motion.button
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      onClick={handleQuickShare}
      disabled={isSharing}
      className={`
        p-2 rounded-full bg-blue-600 hover:bg-blue-700 text-white
        disabled:opacity-50 disabled:cursor-not-allowed transition-all
        ${className}
      `}
      title="Compartilhar"
    >
      <motion.div
        animate={isSharing ? { rotate: 360 } : {}}
        transition={{ duration: 1, repeat: isSharing ? Infinity : 0, ease: "linear" }}
      >
        <Share2 className="w-4 h-4" />
      </motion.div>
    </motion.button>
  )
}

export default ShareButton