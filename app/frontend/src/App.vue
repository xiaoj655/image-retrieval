<script setup>
import { RouterView } from 'vue-router'
import { Analytics } from '@vercel/analytics/vue'

</script>

<template>
  <Analytics/>
  <div class="w-full bg-gray-200 min-h-[calc(100vh-4rem)] rounded-xl p-4 shadow-xl">
    <RouterView />
  </div>
</template>

<style scoped>
</style>
