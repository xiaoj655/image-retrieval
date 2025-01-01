<script setup>
import { ref, onMounted, watch } from 'vue';

const fileInput = ref(null)
const file = ref(null)
const imageData = ref(null)
const displayData = ref([])
const t = ref(0)

function debounce(fn, delay) {
    let timer = null
    return function(...args) {
        clearTimeout(timer)
        timer = setTimeout(() => {
            fn.apply(this, args)
        }, delay)
    }
}

async function _q(){
    const formData = new FormData()
    formData.append("img", file.value)
    if (!file.value) return

    const ret = await fetch(`https://image-retrieval.bitdancing.com/q?t=${t.value}`, {
        method: "POST",
        body: formData
    })
    let data = await ret.json()
    data = data[0].map(i => ({distance: i.distance, label: i.entity.cn_label, _label: i.entity.en_label.split('#')}))
    displayData.value = data
    console.log(displayData.value)
}

const q = debounce(_q, 200)

function _fileHelper(){
    const fileReader = new FileReader()
    fileReader.readAsDataURL(file.value)
    fileReader.onload = () => {
        imageData.value = fileReader.result
    }
}

const handleFileChange = (e) => {
    file.value = e.target.files[0]
    if(!file.value) return
    q()
    _fileHelper()
}

const uploadArea = ref(null)

const uploadAreaText = ref("点击或拖拽上传(支持jpg, jpeg, png, webp, 最大5MB)")
onMounted(() => {
    uploadArea.value.addEventListener("dragover", (e) => {
        e.preventDefault()
        uploadArea.value.classList.add("bg-gray-400")
        uploadAreaText.value = "现在松开鼠标"
    })
    uploadArea.value.addEventListener("dragleave", (e) => {
        e.preventDefault()
        uploadArea.value.classList.remove("bg-gray-400")
        uploadAreaText.value = "点击或拖拽上传"
    })
    uploadArea.value.addEventListener("drop", (e) => {
        e.preventDefault()
        uploadArea.value.classList.remove("bg-gray-400")
        uploadAreaText.value = "点击或拖拽上传"
        file.value = e.dataTransfer.files[0]
        _fileHelper()
        q()
    })
})

function open(url){
    window.open(url, '_blank')
}

const shift = ref('translateX(0)')

watch(t, () => {
    if(t.value === 0){
        shift.value = 'translateX(0)'
    }else{
        shift.value = 'translateX(100%)'
    }
    q()
})
</script>


<template>
<div class="flex flex-col gap-4">
    <div class="text-2xl font-bold flex items-center gap-2">
        基于Resnet预训练模型的电影图像检索
        <a href="">
        <svg viewBox="0 0 24 24" aria-hidden="true" class="size-6 fill-slate-900"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.463 2 11.97c0 4.404 2.865 8.14 6.839 9.458.5.092.682-.216.682-.48 0-.236-.008-.864-.013-1.695-2.782.602-3.369-1.337-3.369-1.337-.454-1.151-1.11-1.458-1.11-1.458-.908-.618.069-.606.069-.606 1.003.07 1.531 1.027 1.531 1.027.892 1.524 2.341 1.084 2.91.828.092-.643.35-1.083.636-1.332-2.22-.251-4.555-1.107-4.555-4.927 0-1.088.39-1.979 1.029-2.675-.103-.252-.446-1.266.098-2.638 0 0 .84-.268 2.75 1.022A9.607 9.607 0 0 1 12 6.82c.85.004 1.705.114 2.504.336 1.909-1.29 2.747-1.022 2.747-1.022.546 1.372.202 2.386.1 2.638.64.696 1.028 1.587 1.028 2.675 0 3.83-2.339 4.673-4.566 4.92.359.307.678.915.678 1.846 0 1.332-.012 2.407-.012 2.734 0 .267.18.577.688.48 3.97-1.32 6.833-5.054 6.833-9.458C22 6.463 17.522 2 12 2Z"></path></svg>
        </a>
    </div>
    <div class="border-b-2 border-black">
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Suscipit, tenetur! Cum impedit quisquam dicta aperiam iusto deleniti omnis veritatis ab quas repellat! Enim aut ea, maiores molestias necessitatibus ab vitae?
        Expedita in enim ducimus ea quas corrupti. Earum est dolores sint reiciendis nemo consequatur laboriosam neque iste? Deserunt blanditiis exercitationem accusamus ducimus commodi corrupti ipsum? In excepturi minus eos exercitationem?
    </div>
    <div class="text-2xl font-bold">
        上传图像
    </div>

    <div @click="fileInput.click()" ref="uploadArea" class="hover:cursor-pointer bg-gray-300 rounded-xl p-4 w-full h-64 flex items-center justify-center flex-col gap-4 relative transition-all">
        <input type="file" accept="image/*" class="hidden" ref="fileInput" @change="handleFileChange"/>

        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-12">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
        </svg>

        {{uploadAreaText}}

        <img :src="imageData" alt="" class="absolute top-0 left-0 w-auto h-full rounded-xl" :class="{'opacity-0': !imageData}">
    </div>

    <div
        class="flex"
    >
        <div class="text-2xl font-bold">结果</div>
        <div class="flex flex-1 justify-center">
            <div class="flex rounded-lg text-sm lg:text-base" :style="{'--transform': shift}">
                <div
                    @click="t = 0"
                 class="px-2 lg:px-6 py-2 rounded-lg hover:bg-violet-200 transition-all hover:cursor-pointer"
                    id="origin">原始模型</div>
                <div 
                    @click="t = 1"
                 class="px-2 lg:px-6 py-2 rounded-lg hover:bg-violet-200 transition-all hover:cursor-pointer">微调之后</div>
            </div>
        </div>
    </div>
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div v-for="item in displayData" class="flex flex-col gap-4">
        <table class="text-center border-separate border-spacing-x-4 border-black border-2 w-full rounded-lg text-sm lg:text-base">
            <tr>
                <td>余弦相似度</td>
                <td>标签</td>
            </tr>
            <tr>
                <td>{{ item.distance.toFixed(4) }}</td>
                <td>{{ item.label.slice(0, -4) }}</td>
            </tr>
        </table>
        <img :src="`https://image-retrieval.bitdancing.com/i/${item._label.join('/')}`" alt="" class="w-full h-auto rounded-lg hover:cursor-pointer" @click="open(`https://image-retrieval.bitdancing.com/i/${item._label.join('/')}`)">
        </div>
    </div>
</div>
</template>

<style scoped>
#origin {
    position: relative;
    &::before{
        content: '';
        z-index: 1;
        border-radius: 0.5rem;
        position: absolute;
        inset: 0;
        background-color: blue;
        opacity: 0.3;
        transition: 0.2s ease;
        transform: var(--transform);
    }
}
</style>