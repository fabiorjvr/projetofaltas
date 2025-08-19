import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { WifiOff, Wifi, CloudOff, RefreshCw } from 'lucide-react'
import { usePWA } from '../../hooks/usePWA'

interface OfflineIndicatorProps {
  className?: string
}

const OfflineIndicator: React.FC<OfflineIndicatorProps> = ({ className = '' }) => {
  const { isOnline } = usePWA()

  return (
    <AnimatePresence>
      {!isOnline && (
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          className={`fixed top-0 left-0 right-0 z-50 ${className}`}
        >
          <div className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-4 py-3 shadow-lg">
            <div className="flex items-center justify-center space-x-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <WifiOff className="w-5 h-5" />
              </motion.div>
              <div className="text-center">
                <p className="font-medium text-sm">
                  Você está offline
                </p>
                <p className="text-xs text-orange-100">
                  Alguns recursos podem estar limitados
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// Connection Status Badge Component
interface ConnectionStatusProps {
  showLabel?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ 
  showLabel = false, 
  size = 'md',
  className = '' 
}) => {
  const { isOnline } = usePWA()
  
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  }
  
  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <motion.div
        animate={isOnline ? { scale: [1, 1.1, 1] } : { opacity: [1, 0.5, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
        className={`${isOnline ? 'text-green-500' : 'text-red-500'}`}
      >
        {isOnline ? (
          <Wifi className={sizeClasses[size]} />
        ) : (
          <WifiOff className={sizeClasses[size]} />
        )}
      </motion.div>
      
      {showLabel && (
        <span className={`${textSizeClasses[size]} font-medium ${
          isOnline ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
        }`}>
          {isOnline ? 'Online' : 'Offline'}
        </span>
      )}
    </div>
  )
}

// Offline Banner Component
interface OfflineBannerProps {
  onRetry?: () => void
  className?: string
}

export const OfflineBanner: React.FC<OfflineBannerProps> = ({ 
  onRetry,
  className = '' 
}) => {
  const { isOnline } = usePWA()
  const [isRetrying, setIsRetrying] = React.useState(false)

  const handleRetry = async () => {
    if (onRetry) {
      setIsRetrying(true)
      try {
        await onRetry()
      } finally {
        setIsRetrying(false)
      }
    } else {
      // Default retry behavior - reload page
      window.location.reload()
    }
  }

  return (
    <AnimatePresence>
      {!isOnline && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className={`bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 border-l-4 border-orange-500 ${className}`}
        >
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg">
                  <CloudOff className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                </div>
                <div>
                  <h3 className="font-medium text-orange-900 dark:text-orange-100">
                    Conexão perdida
                  </h3>
                  <p className="text-sm text-orange-700 dark:text-orange-300">
                    Verifique sua conexão com a internet. Dados em cache ainda estão disponíveis.
                  </p>
                </div>
              </div>
              
              <button
                onClick={handleRetry}
                disabled={isRetrying}
                className="flex items-center space-x-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-orange-400 text-white rounded-lg transition-colors"
              >
                <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                <span className="text-sm font-medium">
                  {isRetrying ? 'Tentando...' : 'Tentar novamente'}
                </span>
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// Sync Status Component
interface SyncStatusProps {
  isSyncing?: boolean
  lastSyncTime?: Date
  className?: string
}

export const SyncStatus: React.FC<SyncStatusProps> = ({ 
  isSyncing = false,
  lastSyncTime,
  className = '' 
}) => {
  const { isOnline } = usePWA()
  
  const formatLastSync = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return 'Agora mesmo'
    if (minutes < 60) return `${minutes}m atrás`
    
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h atrás`
    
    const days = Math.floor(hours / 24)
    return `${days}d atrás`
  }

  return (
    <div className={`flex items-center space-x-2 text-sm ${className}`}>
      <motion.div
        animate={isSyncing ? { rotate: 360 } : {}}
        transition={{ duration: 1, repeat: isSyncing ? Infinity : 0, ease: "linear" }}
        className={`${
          isSyncing 
            ? 'text-blue-500' 
            : isOnline 
              ? 'text-green-500' 
              : 'text-gray-400'
        }`}
      >
        <RefreshCw className="w-4 h-4" />
      </motion.div>
      
      <span className="text-gray-600 dark:text-gray-400">
        {isSyncing ? (
          'Sincronizando...'
        ) : isOnline ? (
          lastSyncTime ? `Sincronizado ${formatLastSync(lastSyncTime)}` : 'Sincronizado'
        ) : (
          'Aguardando conexão'
        )}
      </span>
    </div>
  )
}

export default OfflineIndicator