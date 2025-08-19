import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { 
  Target, 
  TrendingUp, 
  AlertTriangle, 
  Shield,
  Brain,
  Filter,
  ChevronLeft,
  ChevronRight,
  Calendar,
  User
} from 'lucide-react'
import { format } from 'date-fns'
import { ptBR } from 'date-fns/locale'
import { fetchPredictions } from '../services/api'
import type { Prediction } from '../services/api'

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

const predictionTypeColors = {
  foul: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
  yellow_card: 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400',
  red_card: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
}

const predictionTypeLabels = {
  foul: 'Falta',
  yellow_card: 'Cartão Amarelo',
  red_card: 'Cartão Vermelho'
}

const predictionTypeIcons = {
  foul: AlertTriangle,
  yellow_card: Shield,
  red_card: Shield
}

const confidenceColors = {
  low: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  medium: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
  high: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
}

const confidenceLabels = {
  low: 'Baixa',
  medium: 'Média',
  high: 'Alta'
}

export default function Predictions() {
  const [page, setPage] = useState(1)
  const [typeFilter, setTypeFilter] = useState('')
  const [confidenceFilter, setConfidenceFilter] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['predictions', page, typeFilter, confidenceFilter],
    queryFn: () => fetchPredictions({
      page,
      limit: 12
    }),
    keepPreviousData: true
  })

  const formatPredictionDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return format(date, "dd/MM/yyyy 'às' HH:mm", { locale: ptBR })
    } catch {
      return dateString
    }
  }

  const getProbabilityColor = (probability: number) => {
    if (probability >= 80) return 'text-red-600 dark:text-red-400'
    if (probability >= 60) return 'text-orange-600 dark:text-orange-400'
    if (probability >= 40) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-green-600 dark:text-green-400'
  }

  const getProbabilityBg = (probability: number) => {
    if (probability >= 80) return 'bg-red-100 dark:bg-red-900/20'
    if (probability >= 60) return 'bg-orange-100 dark:bg-orange-900/20'
    if (probability >= 40) return 'bg-yellow-100 dark:bg-yellow-900/20'
    return 'bg-green-100 dark:bg-green-900/20'
  }

  const clearFilters = () => {
    setTypeFilter('')
    setConfidenceFilter('')
    setPage(1)
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Erro ao carregar predições
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
      <motion.div variants={itemVariants}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Predições
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Predições de faltas e cartões baseadas em machine learning
        </p>
      </motion.div>

      {/* Summary Stats */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total de Predições
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.total || 0}
              </p>
            </div>
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Target className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Alta Confiança
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.predictions.filter(p => p.confidence === 'high').length || 0}
              </p>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <TrendingUp className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Predições de Faltas
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.predictions.filter(p => p.predictionType === 'foul').length || 0}
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
                Predições de Cartões
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {data?.predictions.filter(p => p.predictionType.includes('card')).length || 0}
              </p>
            </div>
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <Shield className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
          </div>
        </div>
      </motion.div>

      {/* Filters */}
      <motion.div variants={itemVariants} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Filtros
          </h3>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Filter className="h-5 w-5" />
            <span>{showFilters ? 'Ocultar' : 'Mostrar'} Filtros</span>
          </button>
        </div>

        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
          >
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Tipo de Predição
              </label>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="">Todos os tipos</option>
                <option value="foul">Faltas</option>
                <option value="yellow_card">Cartão Amarelo</option>
                <option value="red_card">Cartão Vermelho</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Confiança
              </label>
              <select
                value={confidenceFilter}
                onChange={(e) => setConfidenceFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="">Todas as confianças</option>
                <option value="high">Alta</option>
                <option value="medium">Média</option>
                <option value="low">Baixa</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={clearFilters}
                className="w-full px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                Limpar filtros
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Predictions Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 animate-pulse">
              <div className="flex items-center justify-between mb-4">
                <div className="w-20 h-6 bg-gray-200 dark:bg-gray-600 rounded"></div>
                <div className="w-16 h-6 bg-gray-200 dark:bg-gray-600 rounded"></div>
              </div>
              <div className="space-y-3">
                <div className="h-5 bg-gray-200 dark:bg-gray-600 rounded"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-3/4"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <motion.div 
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {data?.predictions.map((prediction) => {
            const TypeIcon = predictionTypeIcons[prediction.predictionType]
            
            return (
              <motion.div
                key={prediction.id}
                variants={itemVariants}
                whileHover={{ scale: 1.02 }}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
              >
                {/* Prediction Header */}
                <div className="flex items-center justify-between mb-4">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    predictionTypeColors[prediction.predictionType]
                  }`}>
                    <TypeIcon className="h-3 w-3 mr-1" />
                    {predictionTypeLabels[prediction.predictionType]}
                  </span>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    confidenceColors[prediction.confidence]
                  }`}>
                    {confidenceLabels[prediction.confidence]}
                  </span>
                </div>

                {/* Player Info */}
                <div className="mb-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <User className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {prediction.playerName}
                    </h3>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {prediction.team}
                  </p>
                </div>

                {/* Probability */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Probabilidade
                    </span>
                    <span className={`text-lg font-bold ${
                      getProbabilityColor(prediction.probability)
                    }`}>
                      {prediction.probability}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        getProbabilityBg(prediction.probability)
                      }`}
                      style={{ width: `${prediction.probability}%` }}
                    ></div>
                  </div>
                </div>

                {/* AI Insight */}
                <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-1">
                    <Brain className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                    <span className="text-sm font-medium text-purple-600 dark:text-purple-400">
                      Insight da IA
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {prediction.confidence === 'high' 
                      ? 'Padrão comportamental identificado com alta precisão'
                      : prediction.confidence === 'medium'
                      ? 'Indicadores sugerem possibilidade moderada'
                      : 'Predição baseada em dados limitados'
                    }
                  </p>
                </div>

                {/* Timestamp */}
                <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
                  <Calendar className="h-3 w-3 mr-1" />
                  <span>Criado em {formatPredictionDate(prediction.createdAt)}</span>
                </div>
              </motion.div>
            )
          })}
        </motion.div>
      )}

      {/* Pagination */}
      {data && data.totalPages > 1 && (
        <motion.div variants={itemVariants} className="flex items-center justify-between">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            Mostrando {((page - 1) * 12) + 1} a {Math.min(page * 12, data.total)} de {data.total} predições
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPage(page - 1)}
              disabled={page === 1}
              className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <span className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              {page} de {data.totalPages}
            </span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page === data.totalPages}
              className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Empty State */}
      {!isLoading && (!data?.predictions || data.predictions.length === 0) && (
        <motion.div variants={itemVariants} className="text-center py-12">
          <Target className="h-16 w-16 mx-auto text-gray-400 dark:text-gray-600 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Nenhuma predição encontrada
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            As predições serão geradas automaticamente conforme novos dados chegam
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}