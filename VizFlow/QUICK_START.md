# VizFlow - Quick Reference Guide

## 🚀 Getting Started in 30 Seconds

```bash
# Navigate to project
cd c:\Users\Kanad\Desktop\VizFlow\VizFlow

# Start dev server (already running at localhost:5174)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linter
npm run lint
```

## 🎯 What's New in v2.0

### ✨ Enhanced Features
1. **Framer Motion Animations** - Smooth, professional motion throughout
2. **React Hot Toast** - Contextual notifications on every action
3. **CSS Animation System** - 15+ keyframe animations
4. **Improved Interactions** - Hover effects, tap feedback, smooth transitions
5. **Better UX** - Clear visual feedback for all user actions

### 📊 Performance
- ⚡ 60fps smooth animations
- 🎯 GPU-accelerated transforms
- ♿ Accessibility-first approach
- 📦 ~40KB additional size (animation libraries)

## 🎬 Animation Examples

### Toolbar Animations
- Logo rotates on hover
- Buttons have spring physics
- Run icon spins during execution
- All buttons fade in with stagger

### Model Visualization Animations
- Layer boxes scale on hover
- Selected layers glow with pulse
- Zoom buttons provide feedback
- Layer inspector slides in smoothly

### CodeEditor Animations
- Console output fades in
- Indicator pulses continuously
- Smooth content transitions

### SplitPane Animations
- Panes fade in on load
- Divider has interactive hover state
- Smooth width transitions during resize

## 🔔 Toast Notifications

All user actions trigger toasts:

```
Run Code    → "🚀 Running model analysis..."
Save Code   → "💾 Code saved!"
Share       → "🔗 Share link copied!"
Download    → "⬇️ Preparing download..."
Select Layer → "📊 [Layer] selected"
Zoom        → "🔍 [Zoom]%"
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `src/App.jsx` | Main component with Toaster |
| `src/components/Toolbar.jsx` | Animated toolbar with buttons |
| `src/components/CodeEditor.jsx` | Code input with animations |
| `src/components/ModelVisualization.jsx` | Layer visualization with zoom |
| `src/components/SplitPane.jsx` | Draggable split layout |
| `src/index.css` | Global styles + 15+ animations |
| `ANIMATION_SYSTEM.md` | Complete animation guide |
| `DESIGN_ENHANCEMENTS.md` | Design documentation |

## 🎨 Customization Quick Tips

### Change Animation Speed
Find `duration: 0.6` and change the number (seconds)

### Change Animation Colors
Replace hex colors like `#3b82f6` with your color

### Move Toast Position
In App.jsx: `<Toaster position="bottom-right" />`

### Disable Animations
User's OS setting for `prefers-reduced-motion` (auto-respected)

## 🧪 Testing

### View Animations in DevTools
1. Press F12 (DevTools)
2. Go to Animations panel
3. Trigger animations to see timeline

### Check Performance
- Open DevTools Performance tab
- Record interactions
- Verify animations run at 60fps

## 📱 Browser Support

✅ Chrome/Edge, Firefox, Safari (Latest versions)

## 🐛 Troubleshooting

### Animations Not Showing?
```bash
# Clear cache and reinstall
rm -r node_modules package-lock.json
npm install
npm run dev
```

### Toasts Not Appearing?
- Check browser console for errors
- Verify Toaster component in App.jsx
- Check toast styling doesn't have `display: none`

### Performance Issues?
- Reduce animation duration (too much can cause lag)
- Check DevTools Performance tab
- Disable some non-critical animations

## 📚 Documentation

- **ANIMATION_SYSTEM.md** - Patterns, customization, advanced usage
- **DESIGN_ENHANCEMENTS.md** - Design system, colors, typography
- **ENHANCEMENT_SUMMARY.md** - Implementation details, statistics
- **README.md** - Project overview and features

## 🔄 Development Workflow

```
Make changes → Auto-save → Dev server reloads → See changes instantly
```

Hot reload is enabled by default with Vite.

## 🎯 Common Tasks

### Add New Animation
1. Create animation in CSS or use Framer Motion motion.div
2. Import motion from 'framer-motion'
3. Wrap component with motion wrapper
4. Define initial, animate, transition props

### Add New Toast
```jsx
import toast from 'react-hot-toast';

toast.success('Message here', {
  duration: 2000,
});
```

### Modify Colors
Edit `tailwind.config.js` or replace hex colors in CSS

### Change Layout
Modify JSX in component files or adjust Tailwind classes

## 📊 Application Architecture

```
App.jsx (Main)
├── Toolbar (Actions)
├── SplitPane (Layout)
│   ├── CodeEditor (Left)
│   └── ModelVisualization (Right)
└── Footer (Info)

+ Toaster (Notifications)
+ useModelParser (Business Logic)
```

## ✅ Feature Checklist

- [x] Modular component architecture
- [x] Framer Motion animations throughout
- [x] React Hot Toast notifications
- [x] CSS animation system
- [x] Responsive design
- [x] Accessibility support
- [x] Dark theme
- [x] Professional styling
- [x] Hot module reloading
- [x] Production ready

## 🚀 Next Steps

### Immediate (Sprint 1)
- [ ] Monaco Editor integration
- [ ] Language Server Protocol
- [ ] Syntax highlighting

### Short-term (Sprint 2)
- [ ] D3.js visualization
- [ ] Export to PNG/SVG/JSON
- [ ] Framework detection

### Long-term (Sprint 3)
- [ ] FastAPI backend
- [ ] WebSocket support
- [ ] Collaboration features

## 📞 Support

For issues or questions:
1. Check ANIMATION_SYSTEM.md
2. Review DESIGN_ENHANCEMENTS.md
3. Look at component source code
4. Check browser console for errors

## 🎉 You're All Set!

VizFlow v2.0 is now:
- ✨ Visually polished
- 🎬 Smoothly animated
- 🔔 Clearly interactive
- ⚡ Performant
- ♿ Accessible
- 📚 Well-documented

**Start with:** `npm run dev` then visit `http://localhost:5174`

---

**Version**: 2.0 (Enhanced Design)  
**Status**: ✅ Production Ready  
**Last Updated**: 2024
