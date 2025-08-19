import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Target,
  Filter,
  Download,
  Calendar,
  Trophy,
  AlertTriangle,
  Shield,
  Activity,
  Zap
} from 'lucide-react'
import { format } from 'date-fns'
import { ptBR } from 'date-fns/locale'
import { fetchAnalytics } from '../services/api'
import type { Analytics as AnalyticsData } from '../services/api'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
}

const chartVariants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: {
      duration: 0.5,
      ease: "easeOut"
    }
  }
}

export default function Analytics() {
  const [timeRange, setTimeRange] = useState('30d')
  const [selectedMetric, setSelectedMetric] = useState('fouls')
  const [showFilters, setShowFilters] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['analytics', timeRange, selectedMetric],
    queryFn: () => fetchAnalytics({
      timeRange,
      metric: selectedMetric
    }),
    keepPreviousData: true
  })
      card_probability: 0.32,
      risk_level: 'Médio'
    },
    {
      player_name: 'Carlos Oliveira',
      team: 'São Paulo',
      foul_probability: 0.89,
      card_probability: 0.67,
      risk_level: 'Muito Alto'
    }
  ]

  const timeRangeOptions = [
    { value: '7d', label: '7 dias' },
    { value: '30d', label: '30 dias' },
    { value: '90d', label: '90 dias' },
    { value: '1y', label: '1 ano' }
  ]

  const metricOptions = [
    { value: 'fouls', label: 'Faltas', icon: AlertTriangle },
    { value: 'cards', label: 'Cartões', icon: Shield },
    { value: 'performance', label: 'Performance', icon: TrendingUp },
    { value: 'predictions', label: 'Predições', icon: Target }
  ]

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return format(date, 'dd/MM', { locale: ptBR })
    } catch {
      return dateString
    }
  }

  const exportData = () => {
    if (!data) return
    
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Data,Valor\n" +
      data.chartData.map(item => `${item.date},${item.value}`).join("\n")
    
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `analytics_${selectedMetric}_${timeRange}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'Muito Alto':
        return 'text-red-700 bg-red-100 dark:bg-red-900 dark:text-red-300'
      case 'Alto':
        return 'text-orange-700 bg-orange-100 dark:bg-orange-900 dark:text-orange-300'
      case 'Médio':
        return 'text-yellow-700 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-300'
      default:
        return 'text-green-700 bg-green-100 dark:bg-green-900 dark:text-green-300'
    }
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Erro ao carregar analytics
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Tente novamente em alguns instantes
          </p>
        </div>
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
      {/* Header */}
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Analytics
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Análise detalhada de dados e tendências
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Filter className="h-5 w-5" />
            <span>Filtros</span>
          </button>
          <button
            onClick={exportData}
            disabled={!data}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Download className="h-5 w-5" />
            <span>Exportar</span>
          </button>
        </div>
      </motion.div>

      {/* Filters */}
      {showFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          variants={itemVariants}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Filtros de Análise
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Período
              </label>
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {timeRangeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Métrica
              </label>
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {metricOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </motion.div>
      )}

      {/* Key Metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total de Faltas
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.totalFouls || 0}
              </p>
              <p className="text-sm text-green-600 dark:text-green-400 mt-1">
                +12% vs período anterior
              </p>
            </div>
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Cartões Emitidos
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.totalCards || 0}
              </p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                +8% vs período anterior
              </p>
            </div>
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <Shield className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Precisão das Predições
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.predictionAccuracy || 0}%
              </p>
              <p className="text-sm text-green-600 dark:text-green-400 mt-1">
                +3% vs período anterior
              </p>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <Target className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Jogadores Analisados
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.totalPlayers || 0}
              </p>
              <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                +15% vs período anterior
              </p>
            </div>
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Users className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main Chart */}
      <motion.div variants={chartVariants} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Tendência de {metricOptions.find(m => m.value === selectedMetric)?.label}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Últimos {timeRangeOptions.find(t => t.value === timeRange)?.label}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-gray-400" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Tempo real</span>
          </div>
        </div>

        {isLoading ? (
          <div className="h-64 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <div className="h-64 flex items-end justify-between space-x-2">
            {data?.chartData?.map((item, index) => {
              const maxValue = Math.max(...(data.chartData?.map(d => d.value) || [1]))
              const height = (item.value / maxValue) * 100
              
              return (
                <motion.div
                  key={index}
                  initial={{ height: 0 }}
                  animate={{ height: `${height}%` }}
                  transition={{ delay: index * 0.1, duration: 0.5 }}
                  className="flex-1 bg-gradient-to-t from-blue-600 to-blue-400 rounded-t-sm min-h-[4px] relative group"
                >
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {item.value} - {formatDate(item.date)}
                  </div>
                </motion.div>
              )
            }) || []}
          </div>
        )}

        <div className="flex justify-between mt-4 text-xs text-gray-500 dark:text-gray-400">
          {data?.chartData?.map((item, index) => (
            <span key={index} className="flex-1 text-center">
              {formatDate(item.date)}
            </span>
          )) || []}
        </div>
      </motion.div>

      {/* Secondary Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Players */}
        <motion.div variants={itemVariants} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Top Jogadores - Faltas
            </h3>
            <Trophy className="h-5 w-5 text-yellow-500" />
          </div>
          
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex items-center space-x-3 animate-pulse">
                  <div className="w-8 h-8 bg-gray-200 dark:bg-gray-600 rounded-full"></div>
                  <div className="flex-1">
                    <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-3/4 mb-1"></div>
                    <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded w-1/2"></div>
                  </div>
                  <div className="w-8 h-4 bg-gray-200 dark:bg-gray-600 rounded"></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {data?.topPlayers?.map((player, index) => (
                <motion.div
                  key={player.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {index + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {player.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {player.team}
                    </p>
                  </div>
                  <div className="flex-shrink-0">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400">
                      {player.fouls}
                    </span>
                  </div>
                </motion.div>
              )) || []}
            </div>
          )}
        </motion.div>

        {/* Team Performance */}
        <motion.div variants={itemVariants} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Performance por Time
            </h3>
            <Zap className="h-5 w-5 text-blue-500" />
          </div>
          
          {isLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="animate-pulse">
                  <div className="flex justify-between mb-2">
                    <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-1/3"></div>
                    <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-8"></div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2"></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {data?.teamPerformance?.map((team, index) => {
                const maxFouls = Math.max(...(data.teamPerformance?.map(t => t.fouls) || [1]))
                const percentage = (team.fouls / maxFouls) * 100
                
                return (
                  <motion.div
                    key={team.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {team.name}
                      </span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {team.fouls}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ delay: index * 0.1 + 0.2, duration: 0.5 }}
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                      ></motion.div>
                    </div>
                  </motion.div>
                )
              }) || []}
            </div>
          )}
        </motion.div>
      </div>

      {/* Insights */}
      <motion.div variants={itemVariants} className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <TrendingUp className="h-6 w-6 text-blue-600 dark:text-blue-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Insights da IA
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">
              Tendência Identificada
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Aumento de 15% nas faltas durante os últimos 30 dias, principalmente em jogos noturnos.
            </p>
          </div>
          <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">
              Recomendação
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Considere ajustar estratégias de arbitragem para partidas com maior probabilidade de conflitos.
            </p>
          </div>
        </div>
      </motion.div>

    </motion.div>
  )
}

export default Analytics