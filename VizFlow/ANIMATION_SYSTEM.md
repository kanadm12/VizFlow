# VizFlow Animation System Guide

## Quick Overview

VizFlow now features a comprehensive animation system with **Framer Motion** (motion library) and **React Hot Toast** (notifications). Every interaction produces smooth, visual feedback.

## Animation Patterns Used

### 1. **Entrance Animations**
These animate when components first load:

```jsx
// Fade in from top
initial={{ opacity: 0, y: -20 }}
animate={{ opacity: 1, y: 0 }}
transition={{ duration: 0.6, ease: 'easeOut' }}

// Slide in from left
initial={{ opacity: 0, x: -20 }}
animate={{ opacity: 1, x: 0 }}
transition={{ duration: 0.5 }}
```

**Where Used:**
- Toolbar (slideDown)
- CodeEditor (slide from left)
- ModelVisualization (fade in)
- Footer (bottom-up)

### 2. **Hover Animations**
Interactive feedback when user hovers over elements:

```jsx
whileHover={{ scale: 1.05 }}
whileTap={{ scale: 0.95 }}
transition={{ type: 'spring', stiffness: 400 }}
```

**Where Used:**
- All buttons (.btn-primary, .btn-icon)
- Layer boxes in visualization
- Sprint text in footer
- Zoom controls

### 3. **Staggered Children**
Sequential animations for multiple elements:

```jsx
// Parent
variants={containerVariants}
initial="hidden"
animate="visible"

// Children
variants={childVariants}
staggerChildren: 0.1, // 100ms between each
delayChildren: 0.2    // 200ms before first starts
```

**Where Used:**
- Toolbar buttons (appear one by one)
- Form fields
- List items

### 4. **Continuous Animations**
Animations that loop infinitely:

```jsx
// Pulsing
animate={{ scale: [1, 1.2, 1] }}
transition={{ duration: 1.5, repeat: Infinity }}

// Rotating
animate={isRunning ? { rotate: 360 } : { rotate: 0 }}
transition={{ duration: 1, repeat: isRunning ? Infinity : 0 }}
```

**Where Used:**
- Console indicator (pulse)
- Run button icon (rotate during execution)
- Footer text (opacity pulse)

### 5. **Gesture-Based Animations**
Respond to user gestures:

```jsx
whileHover={{ rotate: 10, scale: 1.1 }}
whileTap={{ scale: 0.9 }}
```

**Where Used:**
- Logo icon (hover: rotate + scale)
- All interactive elements

## Toast Notifications

### Setup
```jsx
import toast from 'react-hot-toast';
import { Toaster } from 'react-hot-toast';

// In App.jsx render:
<Toaster
  position="top-right"
  toastOptions={{
    duration: 2000,
    style: {
      background: '#1a1a2e',
      color: '#fff',
      border: '1px solid #3b82f6',
    },
  }}
/>
```

### Usage Examples

**Success Notification:**
```jsx
toast.success('🚀 Running model analysis...', {
  style: {
    background: '#1a1a2e',
    color: '#fff',
    border: '1px solid #3b82f6',
  },
});
```

**Error Notification:**
```jsx
toast.error('❌ Error message', {
  duration: 3000,
});
```

**Custom Notification:**
```jsx
toast((t) => (
  <div>Custom notification</div>
), {
  duration: Infinity,
});
```

### Triggered Toasts in VizFlow

| Action | Message | Icon |
|--------|---------|------|
| Run Code | "🚀 Running model analysis..." | Play |
| Save Code | "💾 Code saved!" | Save |
| Share | "🔗 Share link copied!" | Share |
| Download | "⬇️ Preparing download..." | Download |
| Select Layer | "📊 [Layer Name] selected" | Grid |
| Zoom In/Out | "🔍 [Zoom]%" | Zoom |

## CSS Animation Classes

### Available Classes

**Text Effects:**
- `.text-gradient` - Animated gradient text (3s loop)
- `.glow-text` - Floating text with shadow

**Loading:**
- `.loading-spinner` - Rotating spinner (1s loop)

**Cards & Containers:**
- `.card` - Hover elevation effect
- `.code-editor` - Code area styling
- `.visualization-container` - Graph area styling
- `.glass-morphism` - Blur background effect

**Buttons:**
- `.btn-primary` - Primary blue button with glow
- `.btn-icon` - Small icon buttons

**Badges:**
- `.badge` - Inline badge styling
- `.badge-gradient` - Gradient badge background

### CSS Keyframe Animations

```css
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
  50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
```

## Animation Timing Reference

| Duration | Use Case | Example |
|----------|----------|---------|
| 200ms | Quick feedback | Button hover, icon scale |
| 300ms | Component transition | Panel appearance |
| 500-600ms | Page enter | Toolbar slide down |
| 1000ms+ | Continuous effects | Loading spinner |
| 2-3s | Complex sequences | Gradient shifts, floats |

## Performance Tips

### ✅ Do Use
- Hardware-accelerated properties: `transform`, `opacity`
- Spring animations for natural feel
- Reduced motion media queries for accessibility
- Staggered animations to guide attention

### ❌ Don't Use
- Animate `left`, `top`, `width`, `height` (use `transform` instead)
- Long animations for critical UI (keep under 500ms)
- Too many simultaneous animations
- Animations on page load without intent

## Accessibility Considerations

### Respects User Preferences
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

Users who prefer reduced motion will see instant transitions instead of animations.

### Focus States
All interactive elements have clear focus/hover states for keyboard navigation.

## Common Animation Patterns

### Button with Feedback
```jsx
<motion.button
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
  onClick={handleClick}
>
  Click me
</motion.button>
```

### Fade-in Content
```jsx
<motion.div
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.5 }}
>
  Content
</motion.div>
```

### Staggered List
```jsx
<motion.ul
  initial="hidden"
  animate="visible"
  variants={{
    visible: {
      transition: { staggerChildren: 0.1 }
    }
  }}
>
  {items.map((item) => (
    <motion.li
      key={item}
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0 }
      }}
    >
      {item}
    </motion.li>
  ))}
</motion.ul>
```

## Customization Guide

### Change Animation Speed
In `Toolbar.jsx`:
```jsx
transition={{ duration: 0.6, ease: 'easeOut' }} // Change 0.6 to desired duration
```

### Change Animation Direction
```jsx
// Slide in from right instead of left
initial={{ opacity: 0, x: 20 }} // Changed from x: -20
animate={{ opacity: 1, x: 0 }}
```

### Change Animation Color
In `index.css`:
```css
.text-gradient {
  background: linear-gradient(135deg, #NEW_COLOR, #NEW_COLOR2);
}

.btn-primary {
  background: linear-gradient(to right, #NEW_COLOR, #NEW_COLOR2);
}
```

### Change Toast Position
In `App.jsx`:
```jsx
<Toaster position="bottom-right" /> // Options: top-left, top-center, top-right, etc.
```

## Testing Animations

### Browser DevTools
1. Open DevTools (F12)
2. Go to Animations panel
3. Trigger animations and watch timeline
4. Adjust timing if needed

### Performance Testing
```javascript
// In console
performance.mark('animation-start');
// Wait for animation to complete
performance.mark('animation-end');
performance.measure('animation', 'animation-start', 'animation-end');
```

## Future Enhancement Ideas

1. **Page Transitions**: Animate between different routes
2. **Dark Mode Toggle**: Smooth theme transition animation
3. **Advanced Gestures**: Swipe, pinch animations for mobile
4. **Micro-interactions**: Button press depth effect
5. **Notification Queue**: Stack multiple toasts with animations
6. **Custom Easing**: Define unique bezier curves for specific animations

---

**Last Updated**: 2024  
**Framer Motion Version**: ^10.16.4  
**React Hot Toast Version**: ^2.4.1
