<template>
  <div>
    <div class="mb-4 lg:mb-6">
      <h1 class="text-xl lg:text-2xl font-bold text-gray-900">{{ $t('task.submitTask') }}</h1>
      <p class="mt-1 text-sm text-gray-600">{{ $t('task.processingOptions') }}</p>
    </div>

    <div class="max-w-5xl mx-auto">
      <div class="card mb-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">{{ $t('task.selectFile') }}</h2>
        <FileUploader
          ref="fileUploader"
          :multiple="true"
          :acceptHint="$t('task.supportedFormatsHint')"
          @update:files="onFilesChange"
        />
      </div>

      <div class="card mb-4 lg:mb-6">
        <h2 class="text-base lg:text-lg font-semibold text-gray-900 mb-4">{{ $t('task.processingOptions') }}</h2>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              {{ $t('task.backend') }}
            </label>
            <select
              v-model="config.backend"
              @change="onBackendChange"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="auto">{{ $t('task.backendAuto') }}</option>
              <optgroup :label="$t('task.groupDocParsing')">
                <option value="pipeline">{{ $t('task.backendPipeline') }}</option>
                <option value="paddleocr-vl">{{ $t('task.backendPaddleOCR') }}</option>
                <option value="paddleocr-vl-vllm">{{ $t('task.backendPaddleOCRVLLM') }}</option>
              </optgroup>
              <optgroup :label="$t('task.groupAudioVideo')">
                <option value="sensevoice">{{ $t('task.backendSenseVoice') }}</option>
                <option value="video">{{ $t('task.backendVideo') }}</option>
              </optgroup>
              <optgroup :label="$t('task.groupProfessional')">
                <option value="fasta">{{ $t('task.backendFasta') }}</option>
                <option value="genbank">{{ $t('task.backendGenBank') }}</option>
                <option value="markitdown">MarkItDown (Office)</option>
              </optgroup>
            </select>
            
            <p v-if="config.backend === 'auto'" class="mt-1 text-xs text-gray-500">{{ $t('task.backendAutoHint') }}</p>
            <p v-if="config.backend === 'paddleocr-vl'" class="mt-1 text-xs text-gray-500">{{ $t('task.backendPaddleOCRHint') }}</p>
            <p v-if="config.backend === 'paddleocr-vl-vllm'" class="mt-1 text-xs text-gray-500">{{ $t('task.backendPaddleOCRVLLMHint') }}</p>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              {{ $t('task.language') }}
            </label>
            <select
              v-model="config.lang"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="auto">{{ $t('task.langAuto') }}</option>
              <option value="ch">{{ $t('task.langChinese') }}</option>
              <option value="en">{{ $t('task.langEnglish') }}</option>
              <option value="korean">{{ $t('task.langKorean') }}</option>
              <option value="japan">{{ $t('task.langJapanese') }}</option>
            </select>
            <p class="mt-1 text-xs text-gray-500">
              {{ $t('task.langHint') }}
            </p>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              {{ $t('task.method') }}
            </label>
            <select
              v-model="config.method"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="auto">{{ $t('task.methodAuto') }}</option>
              <option value="txt">{{ $t('task.methodText') }}</option>
              <option value="ocr">{{ $t('task.methodOCR') }}</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              {{ $t('task.priorityLabel') }}
              <span class="text-gray-500 font-normal">{{ $t('task.priorityHint') }}</span>
            </label>
            <input
              v-model.number="config.priority"
              type="number"
              min="0"
              max="100"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        <div v-if="config.backend === 'pipeline'" class="mt-6 space-y-3 pt-4 border-t border-gray-100">
          <label class="flex items-center">
            <input
              v-model="config.formula_enable"
              type="checkbox"
              class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
            />
            <span class="ml-2 text-sm text-gray-700">{{ $t('task.enableFormulaRecognition') }}</span>
          </label>

          <label class="flex items-center">
            <input
              v-model="config.table_enable"
              type="checkbox"
              class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
            />
            <span class="ml-2 text-sm text-gray-700">{{ $t('task.enableTableRecognition') }}</span>
          </label>
        </div>

        <div v-if="['paddleocr-vl', 'paddleocr-vl-vllm'].includes(config.backend)" class="mt-6 pt-6 border-t border-gray-200">
          <h3 class="text-base font-semibold text-blue-900 mb-4 flex items-center">
             PaddleOCR-VL 高级功能
             <span class="ml-2 text-xs font-normal bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">v1.5</span>
          </h3>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-3">
              <h4 class="text-sm font-medium text-gray-700 border-b pb-1 mb-2">图像预处理</h4>
              <label class="flex items-center" title="自动检测并纠正文档旋转角度">
                <input
                  v-model="config.use_doc_orientation_classify"
                  type="checkbox"
                  class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span class="ml-2 text-sm text-gray-700">方向矫正 (处理旋转文档)</span>
              </label>

              <label class="flex items-center" title="自动修复扫描件的弯曲和折痕">
                <input
                  v-model="config.use_doc_unwarping"
                  type="checkbox"
                  class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span class="ml-2 text-sm text-gray-700">扭曲矫正 (处理弯曲扫描件)</span>
              </label>
            </div>

            <div class="space-y-3">
              <h4 class="text-sm font-medium text-gray-700 border-b pb-1 mb-2">识别增强</h4>
              <label class="flex items-center" title="识别并提取文档中的印章">
                <input
                  v-model="config.use_seal_recognition"
                  type="checkbox"
                  class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span class="ml-2 text-sm text-gray-700">印章识别</span>
              </label>

              <label class="flex items-center" title="识别文档中的统计图表">
                <input
                  v-model="config.use_chart_recognition"
                  type="checkbox"
                  class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span class="ml-2 text-sm text-gray-700">图表识别</span>
              </label>
            </div>

            <div class="space-y-3 md:col-span-2">
              <h4 class="text-sm font-medium text-gray-700 border-b pb-1 mb-2">智能排版</h4>
              <div class="flex flex-wrap gap-4">
                <label class="flex items-center" title="将跨页的长表格合并为一个表格">
                  <input
                    v-model="config.merge_tables"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <span class="ml-2 text-sm text-gray-700">合并跨页表格</span>
                </label>

                <label class="flex items-center" title="智能分析并生成 H1/H2/H3 标题层级">
                  <input
                    v-model="config.relevel_titles"
                    type="checkbox"
                    class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <span class="ml-2 text-sm text-gray-700">智能标题分级</span>
                </label>
              </div>
            </div>
            
            <div class="md:col-span-2 mt-2">
               <label class="block text-sm font-medium text-gray-700 mb-1">
                 检测框形状模式
               </label>
               <select
                 v-model="config.layout_shape_mode"
                 class="w-full md:w-1/2 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
               >
                 <option value="auto">自动 (Auto)</option>
                 <option value="rect">矩形 (Rect) - 适合标准文档</option>
                 <option value="poly">多边形 (Poly) - 适合复杂/倾斜文档</option>
               </select>
            </div>
          </div>
        </div>

        <div v-if="config.backend === 'video'" class="mt-6 pt-6 border-t border-gray-200">
          <h3 class="text-base font-semibold text-gray-900 mb-4">{{ $t('task.videoOptions') }}</h3>

          <div class="space-y-4">
            <div>
              <label class="flex items-center">
                <input
                  v-model="config.keep_audio"
                  type="checkbox"
                  class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                />
                <span class="ml-2 text-sm text-gray-700">{{ $t('task.keepAudio') }}</span>
              </label>
              <p class="text-xs text-gray-500 ml-6 mt-1">
                {{ $t('task.keepAudioHint') }}
              </p>
            </div>

            <div class="pt-4 border-t border-gray-100">
              <label class="flex items-center">
                <input
                  v-model="config.enable_keyframe_ocr"
                  type="checkbox"
                  class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                />
                <span class="ml-2 text-sm text-gray-700 font-medium">
                  {{ $t('task.enableKeyframeOCR') }}
                  <span class="ml-1 px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">{{ $t('task.enableKeyframeOCRBadge') }}</span>
                </span>
              </label>
              <p class="text-xs text-gray-500 ml-6 mt-1">
                {{ $t('task.enableKeyframeOCRHint') }}
              </p>

              <div v-if="config.enable_keyframe_ocr" class="ml-6 mt-3 space-y-3 pl-4 border-l-2 border-primary-200">
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-2">
                    {{ $t('task.ocrEngine') }}
                  </label>
                  <select
                    v-model="config.ocr_backend"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="paddleocr-vl">{{ $t('task.ocrEngineRecommended') }}</option>
                  </select>
                </div>

                <label class="flex items-center">
                  <input
                    v-model="config.keep_keyframes"
                    type="checkbox"
                    class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                  />
                  <span class="ml-2 text-sm text-gray-700">{{ $t('task.keepKeyframes') }}</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div v-if="config.backend === 'sensevoice'" class="mt-6 pt-6 border-t border-gray-200">
          <h3 class="text-base font-semibold text-gray-900 mb-4">{{ $t('task.audioOptions') }}</h3>

          <div class="space-y-4">
            <div>
              <label class="flex items-center">
                <input
                  v-model="config.enable_speaker_diarization"
                  type="checkbox"
                  class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                />
                <span class="ml-2 text-sm text-gray-700 font-medium">
                  {{ $t('task.enableSpeakerDiarization') }}
                  <span class="ml-1 px-1.5 py-0.5 text-xs bg-green-100 text-green-700 rounded">{{ $t('task.speakerDiarizationBadge') }}</span>
                </span>
              </label>
              <p class="text-xs text-gray-500 ml-6 mt-1">
                {{ $t('task.speakerDiarizationHint') }}
              </p>
            </div>
          </div>
        </div>

        <div v-if="['pipeline', 'paddleocr-vl', 'paddleocr-vl-vllm'].includes(config.backend)" class="mt-6 pt-6 border-t border-gray-200">
          <h3 class="text-base font-semibold text-gray-900 mb-4">{{ $t('task.watermarkOptions') }}</h3>

          <div class="space-y-4">
            <div>
              <label class="flex items-center">
                <input
                  v-model="config.remove_watermark"
                  type="checkbox"
                  class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                />
                <span class="ml-2 text-sm text-gray-700 font-medium">
                  {{ $t('task.enableWatermarkRemoval') }}
                  <span class="ml-1 px-1.5 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">{{ $t('task.watermarkBadge') }}</span>
                </span>
              </label>
              <p class="text-xs text-gray-500 ml-6 mt-1">
                {{ $t('task.watermarkHint') }}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div v-if="errorMessage" class="card bg-red-50 border-red-200 mb-6">
        <div class="flex items-start">
          <AlertCircle class="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div class="ml-3 flex-1">
            <h3 class="text-sm font-medium text-red-800">{{ $t('common.error') }}</h3>
            <p class="mt-1 text-sm text-red-700">{{ errorMessage }}</p>
          </div>
          <button
            @click="errorMessage = ''"
            class="ml-auto -mr-1 -mt-1 p-1 text-red-600 hover:text-red-800"
          >
            <X class="w-5 h-5" />
          </button>
        </div>
      </div>

      <div class="flex flex-col sm:flex-row justify-end gap-2 sm:gap-3">
        <router-link to="/" class="btn btn-secondary text-center">
          {{ $t('common.cancel') }}
        </router-link>
        <button
          @click="submitTasks"
          :disabled="files.length === 0 || submitting"
          class="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          <Loader v-if="submitting" class="w-4 h-4 mr-2 animate-spin" />
          <Upload v-else class="w-4 h-4 mr-2" />
          {{ submitting ? $t('common.loading') : `${$t('task.submitTask')} (${files.length})` }}
        </button>
      </div>

      <div v-if="submitting || submitProgress.length > 0" class="card mt-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">{{ $t('common.progress') }}</h3>
        <div class="space-y-2">
          <div
            v-for="(progress, index) in submitProgress"
            :key="index"
            class="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
          >
            <div class="flex items-center flex-1">
              <FileText class="w-5 h-5 text-gray-400 flex-shrink-0" />
              <span class="ml-3 text-sm text-gray-900">{{ progress.fileName }}</span>
            </div>
            <div class="flex items-center">
              <CheckCircle v-if="progress.success" class="w-5 h-5 text-green-600" />
              <XCircle v-else-if="progress.error" class="w-5 h-5 text-red-600" />
              <Loader v-else class="w-5 h-5 text-primary-600 animate-spin" />
              <span v-if="progress.taskId" class="ml-2 text-xs text-gray-500">
                {{ progress.taskId }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="!submitting && submitProgress.length > 0" class="mt-4 flex justify-end gap-3">
          <button
            @click="resetForm"
            class="btn btn-secondary"
          >
            {{ $t('common.continue') }}
          </button>
          <router-link to="/tasks" class="btn btn-primary">
            {{ $t('task.taskList') }}
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import FileUploader from '@/components/FileUploader.vue'
import {
  Upload,
  Loader,
  AlertCircle,
  X,
  FileText,
  CheckCircle,
  XCircle,
} from 'lucide-vue-next'
import type { Backend, Language, ParseMethod } from '@/api/types'

const { t } = useI18n()
const router = useRouter()
const taskStore = useTaskStore()

const fileUploader = ref<InstanceType<typeof FileUploader>>()
const files = ref<File[]>([])
const submitting = ref(false)
const errorMessage = ref('')

interface SubmitProgress {
  fileName: string
  success: boolean
  error: boolean
  taskId?: string
}

const submitProgress = ref<SubmitProgress[]>([])

const config = reactive({
  backend: 'auto' as Backend, 
  lang: 'auto' as Language, 
  method: 'auto' as ParseMethod,
  formula_enable: true,
  table_enable: true,
  priority: 0,
  // Video 专属配置
  keep_audio: false,
  enable_keyframe_ocr: false,
  ocr_backend: 'paddleocr-vl',
  keep_keyframes: false,
  // Audio (SenseVoice) 专属配置
  enable_speaker_diarization: false,
  // 水印去除配置
  remove_watermark: false,
  watermark_conf_threshold: 0.35,
  watermark_dilation: 10,
  // [新增] PaddleOCR-VL 专属配置
  use_doc_orientation_classify: false, // 默认关闭，用户手动开启
  use_doc_unwarping: false,            // 默认关闭，用户手动开启
  use_seal_recognition: false,
  use_chart_recognition: false,
  use_ocr_for_image_block: false,
  merge_tables: true,                  // 默认开启
  relevel_titles: true,                // 默认开启
  layout_shape_mode: 'auto'
})

function onFilesChange(newFiles: File[]) {
  files.value = newFiles
}

function onBackendChange() {
  // 根据选择的引擎调整语言设置
  if (config.backend === 'pipeline') {
    // MinerU Pipeline 不支持 auto，默认使用中文
    config.lang = 'ch'
  } else if (['fasta', 'genbank'].includes(config.backend)) {
    // 专业格式引擎不需要语言选择
    config.lang = 'en'
  } else {
    // 其他引擎（auto/音频/视频/OCR）默认自动检测
    config.lang = 'auto'
  }
}

async function submitTasks() {
  if (files.value.length === 0) {
    errorMessage.value = t('task.pleaseSelectFile')
    return
  }

  submitting.value = true
  errorMessage.value = ''
  submitProgress.value = files.value.map(f => ({
    fileName: f.name,
    success: false,
    error: false,
  }))

  // 批量提交任务
  for (let i = 0; i < files.value.length; i++) {
    const file = files.value[i]
    try {
      const response = await taskStore.submitTask({
        file,
        ...config,
      })
      submitProgress.value[i].success = true
      submitProgress.value[i].taskId = response.task_id
    } catch (err: any) {
      submitProgress.value[i].error = true
      console.error(`Failed to submit ${file.name}:`, err)
    }
  }

  submitting.value = false

  // 检查是否全部成功
  const allSuccess = submitProgress.value.every(p => p.success)
  if (allSuccess && files.value.length === 1) {
    // 单个文件且成功，跳转到详情页
    const taskId = submitProgress.value[0].taskId!
    router.push(`/tasks/${taskId}`)
  }
}

function resetForm() {
  files.value = []
  submitProgress.value = []
  errorMessage.value = ''
  fileUploader.value?.clearFiles()
}
</script>
