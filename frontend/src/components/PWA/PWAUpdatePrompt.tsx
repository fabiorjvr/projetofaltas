import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { RefreshCw, X, Zap, Shield } from 'lucide-react'
import { usePWA } from '../../hooks/usePWA'

interface PWAUpdatePromptProps {
  isVisible: boolean
  onClose: () => void
}

const PWAUpdatePrompt: React.FC<PWAUpdatePromptProps> = ({ isVisible, onClose }) => {
  const { updateApp } = usePWA()

  const handleUpdate = async () => {
    await updateApp()
    onClose()
  }

  return (
    <AnimatePresence>
      {isVisible && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            onClick={onClose}
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-md mx-auto bg-white dark:bg-gray-800 rounded-2xl shadow-2xl z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="relative bg-gradient-to-r from-green-600 to-blue-600 p-6 text-white">
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-1 rounded-full hover:bg-white/20 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
              
              <div className="flex items-center space-x-3">
                <div className="p-3 bg-white/20 rounded-full">
                  <RefreshCw className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Nova Versão Disponível</h3>
                  <p className="text-green-100 text-sm">Atualização pronta para instalar</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-6">
              <div className="text-center mb-6">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Atualize para a versão mais recente
                </h4>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  Novas funcionalidades, melhorias de performance e correções de bugs
                </p>
              </div>

              {/* Features */}
              <div className="space-y-4 mb-6">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
                    <Zap className="w-5 h-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white text-sm">
                      Melhor performance
                    </p>
                    <p className="text-gray-500 dark:text-gray-400 text-xs">
                      Carregamento mais rápido e interface otimizada
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                    <Shield className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white text-sm">
                      Correções de segurança
                    </p>
                    <p className="text-gray-500 dark:text-gray-400 text-xs">
                      Patches de segurança e estabilidade
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                    <RefreshCw className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white text-sm">
                      Novas funcionalidades
                    </p>
                    <p className="text-gray-500 dark:text-gray-400 text-xs">
                      Recursos adicionais e melhorias na experiência
                    </p>
                  </div>
                </div>
              </div>

              {/* Update Info */}
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-6">
                <div className="flex items-start space-x-3">
                  <div className="p-1 bg-blue-100 dark:bg-blue-900/50 rounded">
                    <RefreshCw className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">
                      Atualização automática
                    </p>
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      A atualização será aplicada automaticamente. O app será recarregado para aplicar as mudanças.
                    </p>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex space-x-3">
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-3 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-xl font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  Depois
                </button>
                <button
                  onClick={handleUpdate}
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-xl font-medium hover:from-green-700 hover:to-blue-700 transition-all transform hover:scale-105 flex items-center justify-center space-x-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Atualizar Agora</span>
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export default PWAUpdatePrompt