import React from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { useAuthStore } from '../store/authStore'
import { useAppStore } from '../store/useAppStore'
import { AnimatedCard, AnimatedList, LoadingSpinner } from '../components/Animations'
import { Users, Shield, TrendingUp, AlertTriangle } from 'lucide-react'

const Dashboard = () => {
  const { token } = useAuthStore()
  const { addNotification } = useAppStore()

  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/dashboard/stats', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })
      if (!response.ok) throw new Error('Failed to fetch stats')
      return response.json()
    },
    enabled: !!token,
  })

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100
      }
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Carregando dashboard..." />
      </div>
    )
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      <motion.div variants={itemVariants}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Visão geral das estatísticas de faltas e cartões
        </p>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <AnimatedCard delay={0.1} className="p-6">
          <div className="flex items-center">
            <div className="p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Cartões Amarelos</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {stats?.yellow_cards || '1,234'}
              </p>
            </div>
          </div>
        </AnimatedCard>

        <AnimatedCard delay={0.2} className="p-6">
          <div className="flex items-center">
            <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Cartões Vermelhos</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {stats?.red_cards || '156'}
              </p>
            </div>
          </div>
        </AnimatedCard>

        <AnimatedCard delay={0.3} className="p-6">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <TrendingUp className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total de Faltas</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {stats?.total_fouls || '5,678'}
              </p>
            </div>
          </div>
        </AnimatedCard>

        <AnimatedCard delay={0.4} className="p-6">
          <div className="flex items-center">
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <Users className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Jogadores Ativos</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {stats?.active_players || '892'}
              </p>
            </div>
          </div>
        </AnimatedCard>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AnimatedCard delay={0.5} className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Faltas por Mês
          </h3>
          <div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            📊 Gráfico de faltas por mês
            <br />
            (Integração com biblioteca de gráficos)
          </div>
        </AnimatedCard>

        <AnimatedCard delay={0.6} className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Top 10 Jogadores com Mais Faltas
          </h3>
          <div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            📈 Ranking de jogadores
            <br />
            (Lista dos jogadores com mais faltas)
          </div>
        </AnimatedCard>
      </div>

      {/* Recent Activity */}
      <AnimatedCard delay={0.7} className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Atividade Recente
        </h3>
        <AnimatedList stagger={0.1}>
          {[1, 2, 3, 4, 5].map((item) => (
            <div key={item} className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">
                  Novo cartão amarelo registrado para João Silva
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  há {item} minutos
                </p>
              </div>
            </div>
          ))}
        </AnimatedList>
      </AnimatedCard>
    </motion.div>
  )
}

export default Dashboard