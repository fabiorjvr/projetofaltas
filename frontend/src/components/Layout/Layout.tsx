import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Menu, 
  X, 
  Home, 
  Users, 
  Shield, 
  TrendingUp, 
  BarChart3,
  Settings,
  LogOut,
  Sun,
  Moon,
  Download,
  Share2,
  Bell
} from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useTheme } from '../../contexts/ThemeContext'
import { useAuth } from '../../contexts/AuthContext'
import { ConnectionStatus, SyncStatus } from '../PWA'
import { usePWA } from '../../hooks/usePWA'
import ShareButton from '../PWA/ShareButton'
import { useAppStore, useSidebarOpen, useUnreadNotifications } from '../../store/useAppStore'
import { PageTransition } from '../Animations'

interface LayoutProps {
  children: React.ReactNode
}

const menuItems = [
  { path: '/dashboard', label: 'Dashboard', icon: Home },
  { path: '/players', label: 'Jogadores', icon: Users },
  { path: '/teams', label: 'Times', icon: Shield },
  { path: '/predictions', label: 'Predições', icon: TrendingUp },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
]

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const isSidebarOpen = useSidebarOpen()
  const unreadNotifications = useUnreadNotifications()
  const { toggleSidebar, setSidebarOpen } = useAppStore()
  const { theme, toggleTheme } = useTheme()
  const { user, logout } = useAuth()
  const { isInstallable, installApp, isOnline } = usePWA()
  const location = useLocation()
  const [lastSyncTime] = React.useState(new Date())

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and Mobile Menu */}
            <div className="flex items-center space-x-4">
              <motion.button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Menu className="w-6 h-6 text-gray-600 dark:text-gray-400" />
              </motion.button>
              
              <Link to="/dashboard" className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-gray-900 dark:text-white hidden sm:block">
                  Football Analytics
                </span>
              </Link>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex space-x-8">
              {menuItems.map((item) => {
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </nav>

            {/* Right Side Actions */}
            <div className="flex items-center space-x-4">
              {/* Notifications */}
              <motion.button
                className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Bell className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                {unreadNotifications > 0 && (
                  <motion.span
                    className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  >
                    {unreadNotifications > 9 ? '9+' : unreadNotifications}
                  </motion.span>
                )}
              </motion.button>
              
              {/* Connection Status */}
              <div className="hidden sm:flex items-center space-x-3">
                <ConnectionStatus showLabel size="sm" />
                <SyncStatus 
                  isSyncing={false}
                  lastSyncTime={lastSyncTime}
                  className="text-xs"
                />
              </div>

              {/* PWA Install Button */}
              {isInstallable && (
                <button
                  onClick={installApp}
                  className="hidden sm:flex items-center space-x-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                  title="Instalar App"
                >
                  <Download className="w-4 h-4" />
                  <span>Instalar</span>
                </button>
              )}

              {/* Share Button */}
              <ShareButton 
                variant="ghost"
                size="sm"
                showLabel={false}
                className="hidden sm:flex"
                title="Football Fouls Analytics"
                text="Confira esta plataforma de análise de faltas no futebol!"
              />

              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                title={theme === 'dark' ? 'Modo claro' : 'Modo escuro'}
              >
                {theme === 'dark' ? (
                  <Sun className="w-5 h-5 text-yellow-500" />
                ) : (
                  <Moon className="w-5 h-5 text-gray-600" />
                )}
              </button>

              {/* User Menu */}
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">
                    {user?.name?.charAt(0).toUpperCase() || 'U'}
                  </span>
                </div>
                <span className="text-gray-700 dark:text-gray-300 font-medium hidden sm:block">
                  {user?.name || 'Usuário'}
                </span>
              </div>

              {/* Logout Button */}
              <button
                onClick={logout}
                className="p-2 rounded-lg bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors text-red-600 dark:text-red-400"
                title="Sair"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isSidebarOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
              onClick={() => setIsSidebarOpen(false)}
            />
            
            {/* Sidebar */}
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 bottom-0 w-80 bg-white dark:bg-gray-800 shadow-xl z-50 lg:hidden overflow-y-auto"
            >
              {/* Sidebar Header */}
              <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-lg font-bold text-gray-900 dark:text-white">
                    Football Analytics
                  </span>
                </div>
                <button
                  onClick={() => setIsSidebarOpen(false)}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                </button>
              </div>

              {/* Mobile Menu Items */}
              <div className="px-4 py-6 space-y-2">
                {menuItems.map((item) => {
                  const isActive = location.pathname === item.path
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setIsSidebarOpen(false)}
                      className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                      }`}
                    >
                      <item.icon className="w-5 h-5" />
                      <span className="font-medium">{item.label}</span>
                    </Link>
                  )
                })}
              </div>

              {/* Mobile PWA Actions */}
              <div className="px-4 py-4 border-t border-gray-200 dark:border-gray-700 space-y-3">
                {/* Connection Status */}
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Status</span>
                  <ConnectionStatus showLabel size="sm" />
                </div>

                {/* Install App */}
                {isInstallable && (
                  <button
                    onClick={() => {
                      installApp()
                      setSidebarOpen(false)
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    <Download className="w-5 h-5" />
                    <span className="font-medium">Instalar App</span>
                  </button>
                )}

                {/* Share */}
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Compartilhar</span>
                  <ShareButton 
                    variant="ghost"
                    size="sm"
                    showLabel={false}
                    title="Football Fouls Analytics"
                    text="Confira esta plataforma de análise de faltas no futebol!"
                  />
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <PageTransition>
          {children}
        </PageTransition>
      </main>
    </div>
  )
}

export default Layout